import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import fitz  # PyMuPDF
import altair as alt

st.set_page_config(layout="wide")
st.title("CSV-based LLM Query System")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

embedding_model = load_embedding_model()

st.write("Upload your CSV file and ask questions about its content.")

st.markdown("**Note:** To use the LLM, please ensure you have your Groq API key configured in Streamlit's `secrets.toml` file. Create a `.streamlit` folder in your project root and a `secrets.toml` file inside it. Add your key like this: `GROQ_API_KEY=\"your_api_key_here\"`")

uploaded_file = st.file_uploader("Choose a CSV, Excel, PDF, or JSON file", type=["csv", "xlsx", "xls", "pdf", "json"])

import pandas as pd
import json

# ... existing code ...

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'csv':
        st.success("CSV file uploaded successfully!")
        df = pd.read_csv(uploaded_file)
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
    st.session_state['dataframe'] = df
    st.write("### Preview of your CSV file:")
    st.dataframe(df.head())

    # Generate embeddings and build FAISS index
    with st.spinner("Generating embeddings and building FAISS index..."):
        from faiss import IndexHNSWFlat
        import numpy as np

        # Combine all columns into a single text string for each row
        if file_type == 'pdf':
            df['combined_text'] = df['page_content']
        else:
            df['combined_text'] = df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        embeddings = embedding_model.encode(df['combined_text'].tolist())
        dimension = embeddings.shape[1]
        index = IndexHNSWFlat(dimension, 16) # M=16 for HNSW
        index.add(np.array(embeddings).astype('float32'))
        st.session_state['faiss_index'] = index
        st.session_state['data_frame'] = df # Store the DataFrame with combined_text
    st.success("FAISS index built successfully!")
    # Further processing will go here

query = st.text_input("Enter your query here:")

if st.button("Get Answer"):
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    elif not query:
        st.error("Please enter a query.")
    else:
        st.info(f"Processing query: {query}")

        if 'faiss_index' not in st.session_state or 'data_frame' not in st.session_state:
            st.error("Please upload a CSV file and allow the FAISS index to build first.")
        else:
            faiss_index = st.session_state['faiss_index']
            data_frame = st.session_state['data_frame']

            # Hybrid RAG: Retrieve relevant rows based on query embedding
            query_embedding = embedding_model.encode([query])
            D, I = faiss_index.search(query_embedding.astype('float32'), k=5) # Retrieve top 5 relevant rows

            retrieved_rows = data_frame.iloc[I[0]]

            # Prepare context for LLM (Hybrid Row+Parameter Chunking)
            # This example uses a simple concatenation of relevant rows.
            # For more advanced parameter chunking, one might extract specific columns
            # or aggregate data based on the query.
            context = "\n".join(retrieved_rows['combined_text'].tolist())

            # LLM integration will go here, using 'context' and 'query'
            st.session_state['rag_context'] = context
            st.write("Context prepared for LLM:")
            st.code(context)
            # LLM integration
            groq_api_key = st.secrets["GROQ_API_KEY"]
            if not groq_api_key:
                st.error("Groq API key not found. Please add it to your Streamlit secrets.toml file.")
            else:
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
                
                try:
                    chart_spec = json.loads(response_text)
                    if "chart_type" in chart_spec and "x_axis" in chart_spec and "y_axis" in chart_spec:
                        chart_type = chart_spec["chart_type"]
                        x_axis = chart_spec["x_axis"]
                        y_axis = chart_spec["y_axis"]

                        if x_axis not in data_frame.columns or y_axis not in data_frame.columns:
                            st.error(f"Error: X-axis '{x_axis}' or Y-axis '{y_axis}' not found in data.")
                        else:
                            st.write(f"### Generated {chart_type} chart:")
                            if chart_type == "line":
                                chart = alt.Chart(data_frame).mark_line().encode(
                                    x=x_axis,
                                    y=y_axis
                                ).interactive()
                            elif chart_type == "bar":
                                chart = alt.Chart(data_frame).mark_bar().encode(
                                    x=x_axis,
                                    y=y_axis
                                ).interactive()
                            elif chart_type == "scatter":
                                chart = alt.Chart(data_frame).mark_point().encode(
                                    x=x_axis,
                                    y=y_axis
                                ).interactive()
                            else:
                                st.warning(f"Unsupported chart type: {chart_type}. Displaying raw LLM response.")
                                st.write("### Answer:")
                                st.write(response_text)
                            
                            if 'chart' in locals():
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        st.write("### Answer:")
                        st.write(response_text)
                except json.JSONDecodeError:
                    st.write("### Answer:")
                    st.write(response_text)