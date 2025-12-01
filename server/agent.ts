import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { messagesStateReducer, StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";

async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      if (error.status === 429 && attempt < maxRetries - 1) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
        await new Promise((resolve) => setTimeout(resolve, delay));
        continue;
      }
      throw error;
    }
  }
  throw new Error("Max retries exceeded");
}

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
) {
  try {
    const dbName = "inventory_database";
    const db = client.db(dbName);
    const collection = db.collection("items");

    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
      }),
    });

    const itemLookupTool = tool(
      async ({ query, n = 10 }) => {
        try {
          const totalCount = await collection.countDocuments();

          if (totalCount === 0) {
            return JSON.stringify({
              error: "No items found in the database.",
              message: "The items collection is empty.",
              count: 0,
            });
          }

          const dbConfig = {
            collection: collection,
            indexName: "vector_index",
            textKey: "embedding_text",
            embeddingKey: "embedding",
          };

          const vectorStore = new MongoDBAtlasVectorSearch(
            new GoogleGenerativeAIEmbeddings({
              apiKey: process.env.GOOGLE_API_KEY,
              model: "text-embedding-004",
            }),
            dbConfig
          );

          const result = await vectorStore.similaritySearchWithScore(query, n);

          if (result.length === 0) {
            const textResults = await collection
              .find({
                $or: [
                  { item_name: { $regex: query, $options: "i" } },
                  { item_description: { $regex: query, $options: "i" } },
                  { categories: { $regex: query, $options: "i" } },
                  { embedding_text: { $regex: query, $options: "i" } },
                ],
              })
              .limit(n)
              .toArray();

            return JSON.stringify({
              results: textResults,
              searchType: "text",
              query: query,
              count: textResults.length,
            });
          }

          return JSON.stringify({
            results: result,
            searchType: "vector",
            query: query,
            count: result.length,
          });
        } catch (error: any) {
          return JSON.stringify({
            error: "Failed to search inventory",
            details: error.message,
            query: query,
          });
        }
      },
      {
        name: "item_lookup",
        description:
          "Gathers furniture item details from the Inventory database",
        schema: z.object({
          query: z.string(),
          n: z.number().optional().default(10),
        }),
      }
    );

    const tools = [itemLookupTool];
    const toolNode = new ToolNode<typeof GraphState.State>(tools);

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      temperature: 0,
      maxRetries: 0,
      apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools(tools);

    function shouldContinue(state: typeof GraphState.State) {
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1] as AIMessage;
      if (lastMessage.tool_calls?.length) return "tools";
      return "__end__";
    }

    async function callModel(state: typeof GraphState.State) {
      return retryWithBackoff(async () => {
        const prompt = ChatPromptTemplate.fromMessages([
          [
            "system",
            `You are a helpful E-commerce Chatbot Agent for a furniture store.
You must always use the item_lookup tool for furniture searches.
Current time: {time}`,
          ],
          new MessagesPlaceholder("messages"),
        ]);

        const formattedPrompt = await prompt.formatMessages({
          time: new Date().toISOString(),
          messages: state.messages,
        });

        const result = await model.invoke(formattedPrompt);
        return { messages: [result] };
      });
    }

    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue)
      .addEdge("tools", "agent");

    const checkpointer = new MongoDBSaver({ client, dbName });
    const app = workflow.compile({ checkpointer });

    const finalState = await app.invoke(
      { messages: [new HumanMessage(query)] },
      { recursionLimit: 15, configurable: { thread_id } }
    );

    const response =
      finalState.messages[finalState.messages.length - 1].content;
    return response;
  } catch (error: any) {
    if (error.status === 429) {
      throw new Error(
        "Service temporarily unavailable due to rate limits. Please try again in a minute."
      );
    }
    if (error.status === 401) {
      throw new Error(
        "Authentication failed. Please check your API configuration."
      );
    }
    throw new Error(`Agent failed: ${error.message}`);
  }
}
