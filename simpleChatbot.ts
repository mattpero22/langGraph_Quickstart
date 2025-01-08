import dotenv from 'dotenv';
dotenv.config();

const togetherAiKey = process.env.TOGETHER_AI_API_KEY;

import { ChatTogetherAI } from '@langchain/community/chat_models/togetherai';
import { Annotation, MessagesAnnotation } from '@langchain/langgraph';
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { NodeInterrupt } from '@langchain/langgraph';
import { StateGraph } from '@langchain/langgraph';
import { MemorySaver } from '@langchain/langgraph';

const model = new ChatTogetherAI({
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature: 0,
});

const StateAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    nextRepresentative: Annotation<string>,
    refundAuthorized: Annotation<boolean>,
});


// initialize the entrypoint of chatbot and its categorization of requests
// can handle incoming requests and respond or route accordingly to BILLING, TECHNICAL, or RESPOND
// based on the output of nextRepresentative, the model with respond, or route to a corresponding node
const initialSupport = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE = `
    You are frontline support staff for LangCorp, a company that sells computers.
    Be concise in your responses.
    You can chat with customers and help them with basic questions, but if the customer is having a billing or technical issue, do not try and answer the question directly or gather information.
    Instead, immediately transfer them to the billing or technical team by asking them to hold for a moment.
    Otherwise, just respond conversationally.`;
    const supportResponse = await model.invoke([
        { role: "system", content: SYSTEM_TEMPLATE },
        ...state.messages,
    ]);

    const CATEGORIZATION_SYSTEM_TEMPLATE = `
    You are an expert customer support routing system.
    Your job is to detect whether a customer support representative is routing a user to a billing team or a technical team, or if ther are just responding conversationally.`;
    const CATEGORIZATION_HUMAN_TEMPLATE = `
    The previous conversation is an interaction between a customer support representative and a user.
    Extract whether the representative is routing the user to a billing or technical team, or whether they are just responding conversationally.
    Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values: 
        If they want to route the user to the billing team, respond only with the word "BILLING".
        If they want to route the user to the technical team, respond only with the word "TECHNICAL".
        Otherwise, respond only with the word "RESPOND".`;
    const categorizationResponse = await model.invoke([
        { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
        ...state.messages,
        { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE }
    ],
        {
            response_format: {
                type: "json_object",
                schema: zodToJsonSchema(
                    z.object({
                        nextRepresentative: z.enum(["BILLING", "TECHNICAL", "RESPOND"]),
                    })
                )
            }
        }
    );

    const categorizationOutput = JSON.parse(categorizationResponse.content as string);
    return { messages: [supportResponse], nextRepresentative: categorizationOutput.nextRepresentative };
};


const billingSupport = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE = `
    You are an expert billing support specialist for LangCorp, a company that sells computers.
    Help the user to the best of your ability, but be concise in your responses.
    You have the ability to authorize refunds, which you can do by transferring the user to another agent who will collect the required information.
    If you do, assume the other agent has all necessary information about the customer and their order.
    You do not need to ask the user for more information.
    Help the user to the best of your ability, but be concise in your responses.
    `
    let trimmedHistory = state.messages;
    // Make the user's question the most recent message in the history, helping the model stay focused
    if (trimmedHistory.at(-1)?.getType() === "ai") {
        trimmedHistory = trimmedHistory.slice(0, -1);
    };

    const billingRepResponse = await model.invoke([
        { role: "system", content: SYSTEM_TEMPLATE },
        ...trimmedHistory,
    ]);

    const CATEGORIZATION_SYSTEM_TEMPLATE = `
        Your job is to detect whether a billing support representative wants to refund the user.`;
    const CATEGORIZATION_HUMAN_TEMPLATE = `
        The following text is from a customer support representative.
        Extract whether they want to refund the user or not.
        Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values: 
            If they watn to refund the user, respond only with the word "REFUND".
            Otherwise, respond only with the word "RESPOND".
            
        Here is the text:
        <text>
        ${billingRepResponse.content}
        <text>`;
    const categorizationResponse = await model.invoke([
        { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
        { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE }
    ],
        {
            response_format: {
                type: "json_object",
                schema: zodToJsonSchema(
                    z.object({
                        nextRepresentative: z.enum(["REFUND", "RESPOND"])
                    })
                )
            }
        }
    );

    const categorizationOutput = JSON.parse(categorizationResponse.content as string);
    return {
        messages: billingRepResponse,
        nextRepresentative: categorizationOutput.nextRepresentative,
    };
};


const technicalSupport = async (state: typeof StateAnnotation.State) => {
    const SYSTEM_TEMPLATE = `
    You are an expert at diagnosing technical computer issues. You work for a company called LangCorp that sells computers.
    Help the user to the best of your ability, but be concise in your responses.`;

    let trimmedHistory = state.messages;
    if (trimmedHistory.at(-1)?.getType() === "ai") {
        trimmedHistory = trimmedHistory.slice(0, -1);
    };

    const response = await model.invoke([
        {
            role: "system", content: SYSTEM_TEMPLATE
        },
        ...trimmedHistory,
    ]);

    return {
        messages: response,
    };
};


const handleRefund = async (state: typeof StateAnnotation.State) => {
    if (!state.refundAuthorized) {
        console.log("--- HUMAN AUTHORIZATION REQUIRED FOR REFUND ---");
        throw new NodeInterrupt("Human authorization required.");
    }
    return {
        messages: {
            role: "assistant",
            content: "Refund processed!",
        },
    };
};


// Once all the nodes are defined and structured, we can build out the StateGraph
let builder = new StateGraph(StateAnnotation)
    .addNode("initial_support", initialSupport)
    .addNode("billing_support", billingSupport)
    .addNode("technical_support", technicalSupport)
    .addNode("handle_refund", handleRefund)
    .addEdge("__start__", "initial_support")


// after it is initialized with its starting edge, we can define the conditional edges to route to specific nodes based on the response of the current node
builder = builder.addConditionalEdges("initial_support",
    async (state: typeof StateAnnotation.State) => {
        if (state.nextRepresentative.includes("BILLING")) {
            return "billing";
        } else if (state.nextRepresentative.includes("TECHNICAL")) {
            return "technical";
        } else {
            return "conversational";
        }
    },
    {
        billing: "billing_support",
        technical: "technical_support",
        conversational: "__end__",
    }
);
console.log("Added edges!")

// To finish up our edge mapping, we can add an edge to note that technical support should lead to __end__ and that the billing support can either handle_refund then __end__ or just __end__
builder = builder
    .addEdge("technical_support", "__end__")
    .addConditionalEdges("billing_support", async (state) => {
        if (state.nextRepresentative.includes("REFUND")) {
            return "refund";
        } else {
            return "__end__";
        }
    },
        {
            refund: "handle_refund",
            __end__: "__end__"
        })
    .addEdge("handle_refund", "__end__")


// Final step is to .compile() the StateGraph, and store state with an in-memory checkpointer
const checkpointer = new MemorySaver();

const graph = builder.compile({
    checkpointer,
});

async function runChatbot() {
    const stream =  await graph.stream(
        {
            messages: [
                {
                    role: "user",
                    content: "I've changed my mond and I want a refund for order number 11282818!"
                }
            ]
        },
        {
            configurable: {
                thread_id: "refund_testing_id"
            }
        }
    );

    for await (const value of stream) {
        console.log("---STEP---");
        console.log(value);
        console.log("---END STEP ---");
    }
}

runChatbot().catch(error => {
    console.error("Error running agent:", error);
  });



