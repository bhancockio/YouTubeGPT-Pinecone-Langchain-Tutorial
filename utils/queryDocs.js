// https://js.langchain.com/docs/modules/chains/index_related_chains/conversational_retrieval

import { PineconeClient } from "@pinecone-database/pinecone";
import * as dotenv from "dotenv";
import {
  ConversationalRetrievalQAChain,
  VectorDBQAChain,
} from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";

dotenv.config();

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

const vectorStore = await PineconeStore.fromExistingIndex(
  new OpenAIEmbeddings(),
  { pineconeIndex }
);

/* Use as part of a chain (currently no metadata filters) */
const model = new OpenAI();
const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  vectorStore.asRetriever()
);

const question = "What did Alex say about best way to close a sale?";
const res = await chain.call({ question, chat_history: [] });
console.log(res);
/* Ask it a follow up question */
const chatHistory = question + res.text;
const followUpRes = await chain.call({
  question: "Anything else?",
  chat_history: chatHistory,
});
console.log(followUpRes);

//TODO: Create a simple chat app to go back and forth.
