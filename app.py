import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF
import altair as alt

st.set_page_config(layout="wide")
st.title("CSV-based LLM Query System")

MAX_FILE_SIZE_MB = 50

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

embedding_model = load_embedding_model()

st.write("Upload your CSV file and ask questions about its content.")

st.markdown("**Note:** This app requires a Groq API key. Configure it in `.streamlit/secrets.toml`. [Learn more](https://docs.streamlit.io/develop/api-reference/connections/st.secrets)")

uploaded_file = st.file_uploader("Choose a CSV, Excel, PDF, or JSON file", type=["csv", "xlsx", "xls", "pdf", "json"])

import pandas as pd
import json

if uploaded_file is not None:
    # File size validation
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB")
        st.stop()

    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            st.success("CSV file uploaded successfully!")
            df = pd.read_csv(uploaded_file, nrows=100000)
        elif file_type in ['xlsx', 'xls']:
            st.success("Excel file uploaded successfully!")
            excel_file = pd.ExcelFile(uploaded_file)
            df = pd.concat([pd.read_excel(excel_file, sheet_name=sheet)
                           for sheet in excel_file.sheet_names],
                          ignore_index=True)
        elif file_type == 'pdf':
            st.success("PDF file uploaded successfully!")
            def extract_text_from_pdf(pdf_file):
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                text_content = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_content.append({"page_num": page_num + 1, "page_content": page.get_text()})
                return pd.DataFrame(text_content)
            df = extract_text_from_pdf(uploaded_file)
        elif file_type == 'json':
            st.success("JSON file uploaded successfully!")
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to parse {file_type.upper()} file: {str(e)[:200]}")
        st.stop()

    st.session_state['dataframe'] = df
    st.write("### Preview of your file:")
    st.dataframe(df.head())

    # Generate embeddings and build FAISS index
    with st.spinner("Generating embeddings and building FAISS index..."):
        from faiss import IndexFlatL2
        import numpy as np

        # Combine all columns into a single text string for each row
        if file_type == 'pdf':
            df['combined_text'] = df['page_content']
        else:
            df['combined_text'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        embeddings = embedding_model.encode(df['combined_text'].tolist())
        dimension = embeddings.shape[1]
        index = IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        st.session_state['faiss_index'] = index
        st.session_state['dataframe'] = df  # Store the DataFrame with combined_text
    st.success("FAISS index built successfully!")

query = st.text_input("Enter your query here:")

if st.button("Get Answer"):
    if uploaded_file is None:
        st.error("Please upload a file first.")
    elif not query:
        st.error("Please enter a query.")
    else:
        st.info(f"Processing query: {query}")

        if 'faiss_index' not in st.session_state or 'dataframe' not in st.session_state:
            st.error("Please upload a file and allow the FAISS index to build first.")
        else:
            faiss_index = st.session_state['faiss_index']
            data_frame = st.session_state['dataframe']

            # Hybrid RAG: Retrieve relevant rows based on query embedding
            query_embedding = embedding_model.encode([query])
            D, I = faiss_index.search(query_embedding.astype('float32'), k=5)

            retrieved_rows = data_frame.iloc[I[0]]

            context = "\n".join(retrieved_rows['combined_text'].tolist())

            st.session_state['rag_context'] = context
            st.write("Context prepared for LLM:")
            st.code(context)

            # LLM integration with error handling
            try:
                groq_api_key = st.secrets["GROQ_API_KEY"]
            except (KeyError, FileNotFoundError):
                st.error("Groq API key not found. Add `GROQ_API_KEY` to `.streamlit/secrets.toml`.")
                st.stop()

            if not groq_api_key:
                st.error("Groq API key is empty. Please set it in `.streamlit/secrets.toml`.")
            else:
                try:
                    client = Groq(api_key=groq_api_key)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions based on the provided data. If the user asks for a chart or graph, respond with a JSON object containing 'chart_type' (e.g., 'line', 'bar', 'scatter'), 'x_axis', and 'y_axis' fields, and no other text. Otherwise, answer precisely and concisely based on the provided data.",
                            },
                            {
                                "role": "user",
                                "content": f"Based on the following data:\n\n{context}\n\nAnswer the question: {query}",
                            }
                        ],
                        model="llama-3.1-70b-versatile",
                        temperature=0.1,
                        max_tokens=1024,
                        top_p=1,
                        stream=True,
                        stop=None,
                    )

                    response_text = ""
                    for chunk in chat_completion:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
                except Exception as e:
                    st.error(f"LLM API failed: {str(e)[:200]}")
                    st.write("### Raw Context (LLM unavailable):")
                    st.code(context)
                    st.stop()

                # Chart generation with error handling
                try:
                    chart_spec = json.loads(response_text)
                    if "chart_type" in chart_spec and "x_axis" in chart_spec and "y_axis" in chart_spec:
                        chart_type = chart_spec["chart_type"]
                        x_axis = chart_spec["x_axis"]
                        y_axis = chart_spec["y_axis"]

                        if x_axis not in data_frame.columns or y_axis not in data_frame.columns:
                            st.warning(f"Chart axes not found in data. Available: {', '.join(data_frame.columns)}")
                            st.write("### Answer:")
                            st.write(response_text)
                        else:
                            st.write(f"### Generated {chart_type} chart:")
                            try:
                                chart_map = {
                                    "line": alt.Chart(data_frame).mark_line(),
                                    "bar": alt.Chart(data_frame).mark_bar(),
                                    "scatter": alt.Chart(data_frame).mark_point(),
                                }
                                chart_base = chart_map.get(chart_type)
                                if chart_base:
                                    chart = chart_base.encode(x=x_axis, y=y_axis).interactive()
                                    st.altair_chart(chart, use_container_width=True)
                                else:
                                    st.warning(f"Unsupported chart type: {chart_type}. Showing text response.")
                                    st.write("### Answer:")
                                    st.write(response_text)
                            except Exception as e:
                                st.warning(f"Could not generate chart: {e}. Showing text response instead.")
                                st.write("### Answer:")
                                st.write(response_text)
                    else:
                        st.write("### Answer:")
                        st.write(response_text)
                except json.JSONDecodeError:
                    st.write("### Answer:")
                    st.write(response_text)
