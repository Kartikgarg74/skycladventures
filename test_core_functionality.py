from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import fitz  # PyMuPDF

def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def process_dataframe(df, file_path, sheet_name, embedding_model):
    print(f"\nProcessing sheet: {sheet_name}")
    print("Preview of sheet (first 10 rows):")
    print(df.head(10))

    print("Generating embeddings and building FAISS index...")
    # Create combined_text column based on file type
    # Combine all columns into a single 'combined_text' column, handling NaN values
    # and ensuring all values are strings.
    df_temp = df.fillna('').astype(str)
    if file_path.endswith(('.xls', '.xlsx')):
        # For Excel, include sheet_name and drop it from the combined text generation
        df['combined_text'] = df_temp.apply(lambda row: f"Sheet: {row['sheet_name']} " + ' '.join(row.drop('sheet_name', errors='ignore')), axis=1)
    elif file_path.endswith('.pdf'):
        # For PDF, the 'page_content' column already holds the text
        df['combined_text'] = df['page_content']
    else:
        # For CSV, just combine all values
        df['combined_text'] = df_temp.apply(lambda row: ' '.join(row), axis=1)

    # Ensure all combined_text entries are strings
    df['combined_text'] = df['combined_text'].astype(str)




    # Encode embeddings in batches to avoid memory issues
    batch_size = 100 # Adjust batch size as needed
    all_embeddings = []
    for i in range(0, len(df['combined_text']), batch_size):
        batch = df['combined_text'][i:i+batch_size].tolist()
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False, num_workers=0)
        all_embeddings.extend(batch_embeddings)
    embeddings = np.array(all_embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    print("FAISS index built successfully!")
    return {'index': index, 'dataframe': df}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text_by_page.append(page.get_text())
    return text_by_page

def test_faiss_functionality(file_path, query):
    print("Loading embedding model...")
    embedding_model = load_embedding_model()
    print("Embedding model loaded.")

    print(f"Loading file: {file_path}")
    dataframes = {}
    # Heuristic for header detection: look for the first row with mostly non-numeric values
    def find_header_row(df):
        # Check first 10 rows for a header
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            # Count non-numeric values in the row
            non_numeric_count = row.apply(lambda x: not pd.to_numeric(x, errors='coerce') == x).sum()
            # Check if more than 70% of values are non-numeric (a higher threshold)
            if non_numeric_count / len(row) > 0.7:
                return i
            # Also check for common header keywords if the above condition isn't met
            if row.astype(str).str.contains('DRAUGHT|DISP|KMT|VCB|KML|WLA|WSA|CM|CP|CW|LWL|LCA|LCB|MCT|TPC|CB', case=False, na=False).any():
                return i
        return 0 # Default to first row if no clear header found in first 10

    if file_path.endswith('.csv'):
        raw_df = pd.read_csv(file_path, header=None)
        header_row_index = find_header_row(raw_df)
        df = pd.read_csv(file_path, header=header_row_index)

        # Handle duplicate columns after parsing
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(len(cols[cols == dup].index.values.tolist()))]
        df.columns = cols

        # Drop columns that are entirely NaN after setting headers
        df = df.dropna(axis=1, how='all')
        # Drop rows where all values are NaN (empty rows)
        df = df.dropna(axis=0, how='all')
        dataframes['csv_sheet'] = df # Store CSV as a single 'sheet'
    elif file_path.endswith(('.xls', '.xlsx')):
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            raw_df = xls.parse(sheet_name, header=None)
            data_header_row_index = -1
            header_row_index = find_header_row(raw_df)
            df_sheet = xls.parse(sheet_name, header=header_row_index)

            # Handle duplicate columns after parsing
            cols = pd.Series(df_sheet.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(len(cols[cols == dup].index.values.tolist()))]
            df_sheet.columns = cols

            # Drop columns that are entirely NaN after setting headers
            df_sheet = df_sheet.dropna(axis=1, how='all')
            # Drop rows where all values are NaN (empty rows)
            df_sheet = df_sheet.dropna(axis=0, how='all')
            df_sheet['sheet_name'] = sheet_name # Add sheet name column
            dataframes[sheet_name] = df_sheet
    elif file_path.endswith('.pdf'):
        pdf_texts = extract_text_from_pdf(file_path)
        for i, page_text in enumerate(pdf_texts):
            # Create a DataFrame for each page, treating the entire page text as a single row
            df_page = pd.DataFrame([{'page_content': page_text}])
            dataframes[f'pdf_page_{i+1}'] = df_page
    else:
        raise ValueError("Unsupported file type. Please provide a CSV, Excel, or PDF file.")
    print("File loaded successfully!")

    faiss_indexes = {}
    for sheet_name, df in dataframes.items():
        faiss_indexes[sheet_name] = process_dataframe(df, file_path, sheet_name, embedding_model)

    print(f"\nProcessing query: {query}")
    query_embedding = embedding_model.encode([query])

    results = {}
    for sheet_name, data in faiss_indexes.items():
        index = data['index']
        df = data['dataframe']
        D, I = index.search(query_embedding.astype('float32'), k=1) # Retrieve top 1 relevant row, reduced from 5
        retrieved_rows = df.iloc[I[0]]
        context = "\n".join(retrieved_rows['combined_text'].tolist())
        results[sheet_name] = context

    print("\nContexts prepared for LLM (per sheet):")
    for sheet_name, context in results.items():
        print(f"--- Sheet: {sheet_name} ---")
        print(context)


    return results

if __name__ == "__main__":
    # Test with a CSV file
    csv_path = "G-12 BH424-100-006 HYDROSTATIC TABLES.csv"
    test_query_csv = "What is the displacement at 5.0 meters?"
    
    print("\n--- Testing with CSV file ---")
    try:
        results_csv = test_faiss_functionality(csv_path, test_query_csv)
        print("\nFAISS functionality test with CSV completed successfully.")
        for sheet_name, context in results_csv.items():
            print(f"Context for {sheet_name}:\n{context}")
    except Exception as e:
        print(f"\nAn error occurred during FAISS functionality test with CSV: {e}")

    # Test with an Excel file (replace with your actual Excel file path)
    excel_path = "G-12 BH424-100-006 HYDROSTATIC TABLES.xlsx" # Assuming you have this file or create a dummy one
    test_query_excel = "What is the displacement at 5.0 meters?"

    print("\n--- Testing with Excel file ---")
    try:
        results_excel = test_faiss_functionality(excel_path, test_query_excel)
        print("\nFAISS functionality test with Excel completed successfully.")
        for sheet_name, context in results_excel.items():
            print(f"Context for {sheet_name}:\n{context}")
    except Exception as e:
        print(f"\nAn error occurred during FAISS functionality test with Excel: {e}")

    # Test with a PDF file
    pdf_path = "G-12 BH424-100-006 HYDROSTATIC TABLES.pdf"
    test_query_pdf = "What is the displacement at 5.0 meters?"

    print("\n--- Testing with PDF file ---")
    try:
        results_pdf = test_faiss_functionality(pdf_path, test_query_pdf)
        print("\nFAISS functionality test with PDF completed successfully.")
        for page_name, context in results_pdf.items():
            print(f"Context for {page_name}:\n{context}")
    except Exception as e:
        print(f"\nAn error occurred during FAISS functionality test with PDF: {e}")