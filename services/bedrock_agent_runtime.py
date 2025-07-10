import boto3
from botocore.exceptions import ClientError, EventStreamError
import logging
import streamlit as st

logger = logging.getLogger(__name__)

"""
def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    try:
        client = boto3.session.Session().client(service_name="bedrock-agent-runtime",region_name=st.secrets["AWS_DEFAULT_REGION"], aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"], aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
        # See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt
        )

        output_text = ""
        citations = []
        trace = {}

        has_guardrail_trace = False
        for event in response.get("completion"):
            # Combine the chunks to get the output text
            if "chunk" in event:
                chunk = event["chunk"]
                output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations += chunk["attribution"]["citations"]

            # Extract trace information from all events
            if "trace" in event:
                for trace_type in ["guardrailTrace", "preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        mapped_trace_type = trace_type
                        if trace_type == "guardrailTrace":
                            if not has_guardrail_trace:
                                has_guardrail_trace = True
                                mapped_trace_type = "preGuardrailTrace"
                            else:
                                mapped_trace_type = "postGuardrailTrace"
                        if trace_type not in trace:
                            trace[mapped_trace_type] = []
                        trace[mapped_trace_type].append(event["trace"]["trace"][trace_type])

    except (ClientError, EventStreamError) as e:
        logger.error(f"Error invoking agent: {e}")
        return {
            "output_text": "Error: Unable to process agent response. Please try again.",
            "citations": [],
            "trace": {}
        }

    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
"""

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    output_text = ""
    citations = []
    trace = {}
    has_guardrail_trace = False
    try:
        client = boto3.session.Session().client(
            service_name="bedrock-agent-runtime",
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt
        )
        for event in response.get("completion", []):
            logger.info(f"Received Event: {event}")
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations += chunk["attribution"].get("citations", [])
            if "trace" in event:
                for trace_type in ["guardrailTrace", "preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    trace_content = event["trace"].get("trace", {}).get(trace_type)
                    if trace_content:
                        mapped_trace_type = (
                            "preGuardrailTrace" if trace_type == "guardrailTrace" and not has_guardrail_trace
                            else "postGuardrailTrace" if trace_type == "guardrailTrace"
                            else trace_type
                        )
                        has_guardrail_trace = True if trace_type == "guardrailTrace" else has_guardrail_trace
                        trace.setdefault(mapped_trace_type, []).append(trace_content)
    except (ClientError, EventStreamError) as e:
        logger.error("EventStreamError while invoking Bedrock Agent", exc_info=True)
        return {
            "output_text": "Error: Unable to process agent response. Please try again.",
            "citations": [],
            "trace": {}
        }
    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }