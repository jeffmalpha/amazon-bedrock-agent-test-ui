import boto3
from botocore.exceptions import ClientError, EventStreamError, BotoCoreError
import logging
import streamlit as st

logger = logging.getLogger(__name__)

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
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations += chunk["attribution"].get("citations", [])
            if "trace" in event:
                for trace_type in ["guardrailTrace", "preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        mapped_trace_type = trace_type
                        if trace_type == "guardrailTrace":
                            mapped_trace_type = "preGuardrailTrace" if not has_guardrail_trace else "postGuardrailTrace"
                            has_guardrail_trace = True
                        trace.setdefault(mapped_trace_type, []).append(event["trace"]["trace"][trace_type])
    except (EventStreamError, ClientError, BotoCoreError) as e:
        logger.error("Agent call failed", exc_info=True)
        # Extract underlying error message from the exception
        message = str(e)
        if hasattr(e, "response"):
            message = e.response.get("Error", {}).get("Message", str(e))
        # Return this error as a fake assistant message
        return {
            "output_text": f":warning: Agent Error: {message}",
            "citations": [],
            "trace": {}
        }
    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }