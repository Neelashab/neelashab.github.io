import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

import configuration

## Utilities 

# Inspect the tool calls for Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)

## SCHEMA DEFINITIONS

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    fitness_level: Optional[str] = Field(description="Beginner, intermediate, advanced", default=None)
    preferred_workouts: list[str] = Field(description="Workout types user enjoys", default_factory=list)
    injuries: Optional[list[str]] = Field(description="Any injuries or physical limitations", default=[])
    fitness_goals: list[str] = Field(description="Long-term fitness goals", default_factory=list)

# Schema stores user's planned workouts
class WorkoutPlan(BaseModel):
    workout_type: str = Field(description = "Type of workout (e.g., cardio, strength training, yoga)")
    duration: Optional[int] = Field(description = "Estimated time to complete the workout in minutes.")
    intensity: str = Field(description="Intensity level: low, medium, or high.")
    goals: list[str] = Field(description="User's fitness goals for this workout.")
    equipment_needed: Optional[list[str]] = Field(description="List of required equipment.", default=[])


# Schema stores user's completed workouts
class WorkoutHistory(BaseModel):
    date: datetime = Field(description="Date and time of the workout.")
    workout_type: str = Field(description="Type of workout performed.")
    duration: int = Field(description="Actual duration of the workout in minutes.")
    perceived_effort: str = Field(description="User's perceived effort: easy, moderate, hard.")
    feedback: Optional[str] = Field(description="User feedback on the workout.", default=None)



## Initialize the model and tools

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'workout_plan', 'instructions']


# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

## Create the Trustcall extractors for updating the user profile and ToDo list
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)


## Prompts 

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """You are a smart workout assistant.

Your role is to help users **plan workouts**, **track progress**, and **suggest adjustments** based on their fitness level, history, and how they feel on a given day.

You maintain three types of memory:
1. The user's fitness profile (experience, preferences, injuries, goals).
2. The user's workout history (previous workouts, effort level, performance).
3. User-specified preferences for scheduling and modifying workouts.

Based on user input, you should:
- **Suggest a workout for today** based on their fitness level and goals.
- **Modify the workout if requested**, based on how they feel.
- **Track completed workouts** and adjust future suggestions accordingly.

⚡ **If the user says they feel tired or sore, suggest a lower-intensity workout.** 
🔥 **If they feel strong, suggest a higher-intensity option.** 

Here is the current User Profile:
<user_profile>
{user_profile}
</user_profile>

Here is their Workout History:
<workout_history>
{workout_history}
</workout_history>

Here are their preferences:
<instructions>
{instructions}
</instructions>

Respond naturally and be proactive in adapting workouts to the user's needs."""

# TODO -> update this to use tools other than time? 
# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

## Node definitions

def workout_AssIstant(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assert store is not None, "Store is not set in RunnableConfig"
    print(">>>> Store is:", store)


   # Retrieve fitness profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve people memory from the store
    history_namespace = ("workout_history", user_id)
    history_memories = store.search(history_namespace)
    workout_history = "\n".join(f"{mem.value}" for mem in history_memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, 
                                             workout_history=workout_history, 
                                             instructions=instructions)

    # Respond using memory as well as the chat history
    response = model.bind_tools([WorkoutPlan], parallel_tool_calls=False).invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}

def update_workout_plan(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the planned workout memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("workout_plan", user_id)

    # Retrieve the most recent memories for context
    existing_plans = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "WorkoutPlan"
    existing_memories = ([(item.key, tool_name, item.value) 
                          for item in existing_plans] 
                          if existing_plans else None)

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor for updating the planned workouts 
    workout_extractor = create_extractor(
    model,
    tools=[WorkoutPlan],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = workout_extractor.invoke({"messages": updated_messages, 
                                         "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Respond to the tool call made in Workout_AssIstant, confirming the update    
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to Workout_AssIstant
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    # Return tool message with update verification
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig) -> Literal["update_workout_plan", "update_instructions", "update_profile", END]:

    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) ==0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "todo":
            return "update_workout_plan"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions"
        else:
            raise ValueError("Unexpected update_type in tool call")

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(workout_AssIstant)
builder.add_node(update_workout_plan)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow 
builder.add_edge(START, "workout_AssIstant")
builder.add_conditional_edges("workout_AssIstant", route_message)
builder.add_edge("update_workout_plan", "workout_AssIstant")
builder.add_edge("update_profile", "workout_AssIstant")
builder.add_edge("update_instructions", "workout_AssIstant")

# Compile the graph
graph = builder.compile()