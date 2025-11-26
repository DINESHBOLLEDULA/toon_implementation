import streamlit as st
import json
import pandas as pd
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import os
from langsmith import Client
from langsmith.run_helpers import traceable, get_current_run_tree
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="JSON vs TOON Comparison", layout="wide")

load_dotenv()
# Configure Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
LANGSMITH_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", None)
LANGSMITH_PROJECT = "json-vs-toon-comparison"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize LangSmith client
langsmith_client = None
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# Gemini pricing (per 1M tokens)
GEMINI_PRICING = {
    'input': 0.075,
    'output': 0.30
}

def calculate_cost(input_tokens, output_tokens):
    """Calculate cost based on token usage"""
    input_cost = (input_tokens / 1_000_000) * GEMINI_PRICING['input']
    output_cost = (output_tokens / 1_000_000) * GEMINI_PRICING['output']
    return input_cost, output_cost, input_cost + output_cost

def estimate_tokens(text):
    """Estimate tokens (rough approximation: ~4 chars per token)"""
    return len(text) // 4

# Input format selection (ONLY input, no output selection)
st.markdown("## 📄 Input Format Selection")
input_format = st.radio(
    "Select Input Format:",
    ["JSON", "TOON"],
    horizontal=True,
    index=1,  # Default to TOON
    help="This format will be used to send data to the model. Both JSON and TOON outputs will be generated for comparison."
)

# Read CSV file
try:
    df = pd.read_csv('60_Job_Records.csv')
    sample_data = df.to_dict(orient='records')
    
    st.divider()
    
    # Helper function to format as TOON
    def format_as_toon(data):
        try:
            from toon_format import encode
            return encode(data)
        except ImportError:
            # Fallback implementation
            if not data:
                return ""
            if isinstance(data, list) and data:
                columns = list(data[0].keys())
                toon_text = f"[{len(data)},]{{{','.join(columns)}}}:\n"
                for row in data:
                    values = [str(row[col]) for col in columns]
                    toon_text += "  " + ",".join(values) + "\n"
                return toon_text.strip()
            elif isinstance(data, dict):
                toon_text = ""
                for key, value in data.items():
                    toon_text += f"{key}: {value}\n"
                return toon_text.strip()
            return ""
    
    # Display stats based on INPUT format
    st.markdown(f"### 📊 Data Preview ({input_format} Format)")
    col_stats1, col_stats2 = st.columns(2)
    
    with col_stats1:
        if input_format == "JSON":
            json_str = json.dumps(sample_data)
            if GEMINI_API_KEY:
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash-lite')
                    tokens = model.count_tokens(json_str).total_tokens
                except:
                    tokens = estimate_tokens(json_str)
            else:
                tokens = estimate_tokens(json_str)
            st.metric("Estimated Data-Only Tokens", f"{tokens:,}")
        else:
            toon_text = format_as_toon(sample_data)
            if GEMINI_API_KEY:
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash-lite')
                    tokens = model.count_tokens(toon_text).total_tokens
                except:
                    tokens = estimate_tokens(toon_text)
            else:
                tokens = estimate_tokens(toon_text)
            st.metric("Estimated Data-Only Tokens", f"{tokens:,}")

    with col_stats2:
        if input_format == "JSON":
            json_size = len(json.dumps(sample_data))
            st.metric("Input Format Size", f"{json_size:,} chars")
        else:
            toon_text = format_as_toon(sample_data)
            st.metric("Input Format Size", f"{len(toon_text):,} chars")
    
    # Custom CSS
    st.markdown(
        """
        <style>
        .fixed-height-container {
            height: 400px;
            max-width: 800px;
            margin: 0 auto;
            overflow-y: auto;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .fixed-height-container pre {
            margin: 0;
            white-space: pre;
            color: #d4d4d4;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .response-container {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f5f5f5;
            max-height: 500px;
            overflow-y: auto;
        }
        .json-output {
            border-left: 4px solid #2196F3;
        }
        .toon-output {
            border-left: 4px solid #4CAF50;
        }
        .response-title {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .json-title {
            color: #2196F3;
        }
        .toon-title {
            color: #4CAF50;
        }
        .config-box {
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
        }
        .comparison-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .savings-positive {
            color: #4CAF50;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display data based on INPUT format
    if input_format == "JSON":
        json_str = json.dumps(sample_data, indent=2)
        st.markdown(
            f'<div class="fixed-height-container"><pre>{json_str}</pre></div>',
            unsafe_allow_html=True
        )
    else:  # TOON
        toon_text = format_as_toon(sample_data)
        st.markdown(
            f'<div class="fixed-height-container"><pre>{toon_text}</pre></div>',
            unsafe_allow_html=True
        )
    
except FileNotFoundError:
    st.error("❌ CSV file not found. Please make sure '60_Job_Records.csv' exists in the same directory.")
    st.stop()

# Query section
st.divider()
st.markdown("### 🔍 Query Analysis")

# Show current configuration
st.markdown(
    f"""
    <div class="config-box">
        <strong>📋 Test Configuration:</strong> {input_format} input → JSON output vs {input_format} input → TOON output<br>
        <em>Both outputs will be generated simultaneously for comparison</em>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "",
        placeholder="Enter a query to analyze the data...",
        label_visibility="collapsed",
        key="query_input"
    )

with col2:
    analyze_button = st.button("🚀 Analyze", use_container_width=True)

# Helper functions to format data
def format_data_as_json(data):
    """Format data as JSON"""
    return json.dumps(data, indent=2)

def format_data_as_toon(data):
    """Format data as TOON using toon_format library"""
    try:
        from toon_format import encode
        return encode(data)
    except ImportError:
        # Fallback implementation
        if data and isinstance(data, list) and data:
            columns = list(data[0].keys())
            toon_text = f"[{len(data)},]{{{','.join(columns)}}}:\n"
            for row in data:
                values = [str(row[col]) for col in columns]
                toon_text += "  " + ",".join(values) + "\n"
            return toon_text.strip()
        elif data and isinstance(data, dict):
            toon_text = ""
            for key, value in data.items():
                toon_text += f"{key}: {value}\n"
            return toon_text.strip()
        return ""

# Function to get response from Gemini with LangSmith tracking
@traceable(
    run_type="llm",
)
def get_response(data, query, input_fmt, output_fmt):
    """
    Get response from Gemini with specified input and output formats.
    Tracks usage in LangSmith using @traceable decorator.
    """
    try:
        # Define system prompts based on OUTPUT format
        if output_fmt == "JSON":
            system_prompt = """You are an AI that strictly outputs valid JSON and nothing else.
RULES:
1. The assistant output MUST be only valid JSON.
2. No explanations, no markdown, no comments, no text outside JSON.
3. Do not wrap JSON in code blocks or backticks.
4. The JSON must be syntactically valid and parseable.
5. Format the JSON with proper indentation (use 2 spaces per level).
6. If the user provides TOON format input, understand it internally but respond only in JSON.
7. Output must start with [ or { and end with ] or }."""
        else:  # TOON
            system_prompt = """You are an AI that strictly outputs valid TOON format and nothing else.

TOON FORMAT SPECIFICATION:
TOON is a compact, human-readable data format with this syntax:

1. ARRAYS OF OBJECTS (tabular data):
   [count,]{field1,field2,field3}:
     value1,value2,value3
     value1,value2,value3

   Example:
   [2,]{id,name,role}:
     1,Alice,admin
     2,Bob,user

2. SIMPLE OBJECTS (key-value pairs):
   key1: value1
   key2: value2

   Example:
   name: Alice
   age: 30

3. NAMED ARRAYS:
   arrayName[count]: item1,item2,item3

   Example:
   items[2]: apple,banana

RULES:
1. For arrays of objects, use format: [count,]{fields}: followed by comma-separated values
2. For simple objects, use key: value format (one per line)
3. For named arrays, use arrayName[count]: followed by comma-separated items
4. Use 2 spaces for indentation when nested
5. No quotes around strings unless they contain special characters (commas, colons)
6. Do NOT output JSON
7. No markdown, no code blocks, no explanations
8. Do NOT wrap in tags - just output the TOON format directly

Your role: Convert input data to TOON format following the exact syntax above."""
        
        # Format data based on INPUT format
        if input_fmt == "JSON":
            formatted_data = format_data_as_json(data)
        else:
            formatted_data = format_data_as_toon(data)
        
        user_prompt = f"""Input Data ({input_fmt} format):
{formatted_data}

User Query: {query}

Respond with ONLY the {output_fmt} output."""
        
        # Calculate overhead tokens more accurately using Gemini's count_tokens
        temp_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Get actual token counts
        data_only_tokens_actual = temp_model.count_tokens(formatted_data).total_tokens
        system_prompt_tokens = temp_model.count_tokens(system_prompt).total_tokens
        query_wrapper = f"Input Data ({input_fmt} format):\n\nUser Query: {query}\n\nRespond with ONLY the {output_fmt} output."
        query_wrapper_tokens = temp_model.count_tokens(query_wrapper).total_tokens - data_only_tokens_actual
        overhead_tokens = system_prompt_tokens + query_wrapper_tokens
        
        model = genai.GenerativeModel(
            'gemini-2.0-flash-lite',
            system_instruction=system_prompt
        )
        
        response = model.generate_content(user_prompt)
        result = response.text.strip()
        
        # Extract token usage from API
        usage_metadata = response.usage_metadata
        total_input_tokens_api = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        total_tokens_api = usage_metadata.total_token_count
        
        # Use the pre-calculated data-only tokens from Gemini's count_tokens
        # This is more accurate than subtracting estimates
        data_only_input_tokens = data_only_tokens_actual
        
        # Calculate costs using ONLY data tokens for input cost
        data_input_cost = (data_only_input_tokens / 1_000_000) * GEMINI_PRICING['input']
        output_cost = (output_tokens / 1_000_000) * GEMINI_PRICING['output']
        total_cost = data_input_cost + output_cost
        
        # Clean up markdown
        if result.startswith('```'):
            lines = result.split('\n')
            result = '\n'.join(lines[1:-1]) if len(lines) > 2 else result
            result = result.replace('```json', '').replace('```', '').strip()
        
        # Validate JSON if output is JSON
        if output_fmt == "JSON":
            try:
                parsed = json.loads(result)
                result = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
        
        # Prepare usage metadata for LangSmith
        usage_metadata_dict = {
            "input_tokens": data_only_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": data_only_input_tokens + output_tokens,
        }
        
        # Prepare detailed metadata
        metadata = {
            "format": f"{input_fmt}_to_{output_fmt}",
            "ls_provider": "google_genai",
            "ls_model_name": "gemini-2.0-flash-lite",
            "input_format": input_fmt,
            "output_format": output_fmt,
            "total_input_tokens_api": total_input_tokens_api,
            "data_only_input_tokens": data_only_input_tokens,
            "overhead_tokens": overhead_tokens,
            "data_input_cost_usd": round(data_input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "data_size_chars": len(formatted_data),
            "response_size_chars": len(result),
            "query": query,
        }
        
        # Update LangSmith run tree with outputs and usage
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.set(
                outputs={"response": result},
                usage_metadata=usage_metadata_dict,
                metadata=metadata,
            )
            run_tree.name = f"{input_fmt}_to_{output_fmt}"
        
        return result, {
            'total_input_tokens_api': total_input_tokens_api,
            'data_only_input_tokens': data_only_input_tokens,
            'overhead_tokens': overhead_tokens,
            'output_tokens': output_tokens,
            'total_tokens_api': total_tokens_api,
            'data_input_cost': data_input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
            
    except Exception as e:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.set(
                error=str(e),
                outputs={"error": str(e)}
            )
        return f"Error: {str(e)}", None

# Handle query analysis
if analyze_button and query:
    if not GEMINI_API_KEY:
        st.error("❌ Please set GEMINI_API_KEY to use this feature.")
    else:
        with st.spinner("Analyzing with Gemini... Running both JSON and TOON outputs"):
            # Run BOTH outputs simultaneously with the SAME input format
            with ThreadPoolExecutor(max_workers=2) as executor:
                json_future = executor.submit(get_response, sample_data, query, input_format, "JSON")
                toon_future = executor.submit(get_response, sample_data, query, input_format, "TOON")
                
                json_response, json_metrics = json_future.result()
                toon_response, toon_metrics = toon_future.result()
            
            st.success(f"✅ Query executed: **{query}**")
            
            # Comparison Header
            st.markdown(
                f"""
                <div class="comparison-header">
                    <h3 style="margin:0;">📊 Comparison (Same Input, Different Outputs)</h3>
                    <p style="margin:5px 0 0 0;">Test Configuration: {input_format} input → JSON output vs {input_format} input → TOON output</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show data-only input tokens comparison (should be identical)
            if json_metrics and toon_metrics:
                input_token_info = f"JSON Output: {json_metrics['data_only_input_tokens']:,} | TOON Output: {toon_metrics['data_only_input_tokens']:,}"
                if abs(json_metrics['data_only_input_tokens'] - toon_metrics['data_only_input_tokens']) <= 10:
                    st.info(f"ℹ️ **Data-Only Input Tokens:** {input_token_info} (Identical - both use {input_format} input)")
                else:
                    st.warning(f"⚠️ **Data-Only Input Tokens:** {input_token_info} (Difference detected - may need investigation)")
            
            # Side-by-side comparison
            col_json, col_toon = st.columns(2)
            
            with col_json:
                st.markdown('<div class="response-title json-title">📄 JSON Output ({} Input)</div>'.format(input_format), 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="response-container json-output"><pre>{json_response}</pre></div>', 
                           unsafe_allow_html=True)
                
                if json_metrics:
                    st.markdown("**Token Usage & Cost:**")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Data Input Tokens", f"{json_metrics['data_only_input_tokens']:,}")
                        st.metric("Output Tokens", f"{json_metrics['output_tokens']:,}")
                    with metric_col2:
                        st.metric("Overhead Tokens", f"{json_metrics['overhead_tokens']:,}")
                        st.metric("Total Cost", f"${json_metrics['total_cost']:.6f}")
                    
                    with st.expander("💰 Cost Breakdown"):
                        st.write(f"**Total API Input Tokens:** {json_metrics['total_input_tokens_api']:,}")
                        st.write(f"**Data-Only Input Tokens:** {json_metrics['data_only_input_tokens']:,}")
                        st.write(f"**Overhead Tokens (excluded):** {json_metrics['overhead_tokens']:,}")
                        st.write(f"**Data Input Cost:** ${json_metrics['data_input_cost']:.6f}")
                        st.write(f"**Output Cost:** ${json_metrics['output_cost']:.6f}")
                        st.write(f"**Total Cost:** ${json_metrics['total_cost']:.6f}")
            
            with col_toon:
                st.markdown('<div class="response-title toon-title">🎯 TOON Output ({} Input)</div>'.format(input_format), 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="response-container toon-output"><pre>{toon_response}</pre></div>', 
                           unsafe_allow_html=True)
                
                if toon_metrics:
                    st.markdown("**Token Usage & Cost:**")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Data Input Tokens", f"{toon_metrics['data_only_input_tokens']:,}")
                        st.metric("Output Tokens", f"{toon_metrics['output_tokens']:,}")
                    with metric_col2:
                        st.metric("Overhead Tokens", f"{toon_metrics['overhead_tokens']:,}")
                        st.metric("Total Cost", f"${toon_metrics['total_cost']:.6f}")
                    
                    with st.expander("💰 Cost Breakdown"):
                        st.write(f"**Total API Input Tokens:** {toon_metrics['total_input_tokens_api']:,}")
                        st.write(f"**Data-Only Input Tokens:** {toon_metrics['data_only_input_tokens']:,}")
                        st.write(f"**Overhead Tokens (excluded):** {toon_metrics['overhead_tokens']:,}")
                        st.write(f"**Data Input Cost:** ${toon_metrics['data_input_cost']:.6f}")
                        st.write(f"**Output Cost:** ${toon_metrics['output_cost']:.6f}")
                        st.write(f"**Total Cost:** ${toon_metrics['total_cost']:.6f}")
            
            # Show detailed comparison
            st.divider()
            st.markdown("### 📊 Detailed Comparison")
            
            if json_metrics and toon_metrics:
                # Calculate savings
                output_token_diff = json_metrics['output_tokens'] - toon_metrics['output_tokens']
                output_token_savings = (output_token_diff / json_metrics['output_tokens'] * 100) if json_metrics['output_tokens'] > 0 else 0
                
                cost_diff = json_metrics['total_cost'] - toon_metrics['total_cost']
                cost_savings = (cost_diff / json_metrics['total_cost'] * 100) if json_metrics['total_cost'] > 0 else 0
                
                # Summary metrics
                comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                
                with comp_col1:
                    st.metric("JSON Output Size", f"{len(json_response):,} chars")
                with comp_col2:
                    st.metric("TOON Output Size", f"{len(toon_response):,} chars")
                with comp_col3:
                    st.metric("Output Token Savings", f"{output_token_savings:.1f}%", 
                             delta=f"{output_token_diff:,} tokens",
                             delta_color="normal" if output_token_diff > 0 else "inverse")
                with comp_col4:
                    st.metric("Cost Savings", f"{cost_savings:.1f}%", 
                             delta=f"${cost_diff:.6f}",
                             delta_color="normal" if cost_diff > 0 else "inverse")
                
                # Detailed comparison table
                st.markdown("#### Detailed Metrics")
                comparison_df = pd.DataFrame({
                    'Metric': ['Data Input Tokens', 'Overhead Tokens', 'Output Tokens', 'Total API Tokens', 'Data Input Cost', 'Output Cost', 'Total Cost'],
                    f'{input_format}→JSON': [
                        f"{json_metrics['data_only_input_tokens']:,}",
                        f"{json_metrics['overhead_tokens']:,}",
                        f"{json_metrics['output_tokens']:,}",
                        f"{json_metrics['total_tokens_api']:,}",
                        f"${json_metrics['data_input_cost']:.6f}",
                        f"${json_metrics['output_cost']:.6f}",
                        f"${json_metrics['total_cost']:.6f}"
                    ],
                    f'{input_format}→TOON': [
                        f"{toon_metrics['data_only_input_tokens']:,}",
                        f"{toon_metrics['overhead_tokens']:,}",
                        f"{toon_metrics['output_tokens']:,}",
                        f"{toon_metrics['total_tokens_api']:,}",
                        f"${toon_metrics['data_input_cost']:.6f}",
                        f"${toon_metrics['output_cost']:.6f}",
                        f"${toon_metrics['total_cost']:.6f}"
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("#### 💡 Key Insights")
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    if output_token_savings > 0:
                        st.success(f"✅ TOON output saved **{output_token_diff:,} tokens** ({output_token_savings:.1f}%) on output")
                    else:
                        st.info(f"ℹ️ JSON output used {abs(output_token_diff):,} fewer tokens")
                
                with col_insight2:
                    if cost_savings > 0:
                        st.success(f"✅ TOON output saved **${cost_diff:.6f}** ({cost_savings:.1f}%) in total cost")
                    else:
                        st.info(f"ℹ️ JSON output cost ${abs(cost_diff):.6f} less")
                
                # Show methodology note
                st.info("📌 **Cost Calculation Note:** Input costs are calculated using only data tokens (excluding system prompt and query overhead tokens) for accurate format comparison.")
                
                if langsmith_client:
                    st.success(f"✅ Metrics logged to LangSmith project '{LANGSMITH_PROJECT}' with comprehensive token and cost tracking.")
            
elif analyze_button:
    st.warning("⚠️ Please enter a query first.")