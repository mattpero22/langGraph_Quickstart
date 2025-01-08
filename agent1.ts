// agent1.ts

import dotenv from 'dotenv';
dotenv.config();

const OpenAiKey = process.env.OPENAI_API_KEY;
const TavilyKey = process.env.TAVILY_API_KEY;

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Define the tools for the agent to use
const agentTools = [new TavilySearchResults({ maxResults: 3 })];
const agentModel = new ChatOpenAI({ temperature: 0 });

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// Now it's time to use!
async function runAgent() {
    const agentFinalState = await agent.invoke(
      { messages: [new HumanMessage("what is the cost of a gallon of gas in california?")] },
      { configurable: { thread_id: "42" } },
    );
  
    console.log(
      agentFinalState.messages[agentFinalState.messages.length - 1].content,
    );
  
    const agentNextState = await agent.invoke(
      { messages: [new HumanMessage("what about new york?")] },
      { configurable: { thread_id: "42" } },
    );
  
    console.log(
      agentNextState.messages[agentNextState.messages.length - 1].content,
    );
  }


runAgent().catch(error => {
    console.error("Error running agent:", error);
  });