import os
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import gradio as gr

movies=pd.read_csv('movies_with_emotions.csv')

embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db_path='chroma_db'

if os.path.exists(db_path) and os.listdir(db_path):
    db_movies=Chroma(persist_directory=db_path, embedding_function=embedding_model)
    print(f'Loaded existing Chroma DB from: {db_path}')

else:
    raw_documents=TextLoader('tagged_description.txt',encoding='utf-8').load()
    text_splitters=CharacterTextSplitter(chunk_size=1,chunk_overlap=0,separator='\n')
    documents=text_splitters.split_documents(raw_documents)
    
    documents_with_metadata = []
    
    for i, doc in enumerate(documents):
        movie_idx = i % len(movies)
        row = movies.iloc[movie_idx]
    
        new_doc = Document(
            page_content=doc.page_content,
            metadata={
                'name': row['name'],
                'date': row['date'],
                'actor': row['actor'],
                'director': row['director'],
                'language': row['language'],
                }
        )
        
        documents_with_metadata.append(new_doc)
        
    db_movies=Chroma.from_documents(documents_with_metadata,
                                embedding=embedding_model,
                                persist_directory='chroma_db')
    
    print(f"Created new Chroma DB at: {db_path}")    

def enhanced_similarity_search(query:str,
                               genre:str=None,
                               tone:str=None,
                               top_k= 30,
                               final_top_k=10) -> pd.DataFrame:
    metadata_matches = []
    
    # Check if query matches any movie names
    name_matches = movies[movies['name'].str.contains(query, case=False, na=False)]
    for _, movie in name_matches.iterrows():
        docs = db_movies.similarity_search(
            query, 
            k=top_k,
            filter={"name": movie['name']}
        )
        metadata_matches.extend(docs)
    
    # Check if query matches any directors
    director_matches = movies[movies['director'].str.contains(query, case=False, na=False)]
    for _, movie in director_matches.iterrows():
        docs = db_movies.similarity_search(
            query, 
            k=top_k,
            filter={"director": movie['director']}
        )
        metadata_matches.extend(docs)
    
    # Check if query matches any actors
    actor_matches = movies[movies['actor'].str.contains(query, case=False, na=False)]
    for _, movie in actor_matches.iterrows():
        docs = db_movies.similarity_search(
            query, 
            k=top_k,
            filter={"actor": movie['actor']}
        )
        metadata_matches.extend(docs)
    
    # Check if query matches any dates
    date_matches = movies[movies['date'].astype(str).str.contains(query, na=False)]
    for _, movie in date_matches.iterrows():
        docs = db_movies.similarity_search(
            query, 
            k=top_k,
            filter={"date": movie['date']}
        )
        metadata_matches.extend(docs)
    
    # Remove duplicates
    seen_ids = set()
    unique_metadata_matches = []
    for doc in metadata_matches:
        doc_id = f"{doc.metadata['name']}_{doc.page_content[:7]}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_metadata_matches.append(doc)
    
    movies_list = []
    
    # If we have metadata matches, use them
    if unique_metadata_matches:
        for doc in unique_metadata_matches:
            try:
                movie_id = int(doc.page_content.strip('"').split()[0])
                movies_list.append(movie_id)
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Error extracting movie ID from document: {e}")
                continue
    else:
        regular_results = db_movies.similarity_search(query, k=top_k)
        for doc in regular_results:
                movie_id = int(doc.page_content.strip('"').split()[0])
                movies_list.append(movie_id)
    
    movie_recs = movies[movies['id'].isin(movies_list)].head(final_top_k)
    
    if genre!='All':
        movie_recs=movie_recs[movie_recs['genre'].str.contains(genre, case=False, na=False)]
        
    if tone=='Happy':
        movie_recs.sort_values(by='joy',ascending=False,inplace=True)
    elif tone=='Surprising':
        movie_recs.sort_values(by='surprise',ascending=False,inplace=True)
    elif tone=='Angry':
        movie_recs.sort_values(by='anger',ascending=False,inplace=True)
    elif tone=='Suspenseful':
        movie_recs.sort_values(by='fear',ascending=False,inplace=True)
    elif tone=='Sad':
        movie_recs.sort_values(by='sadness',ascending=False,inplace=True)
    elif tone=='Gross':
        movie_recs.sort_values(by='disgust',ascending=False,inplace=True)
        
    return movie_recs

def recommend_movies(query:str, genre:str, tone:str):
    recommendations=enhanced_similarity_search(query,genre,tone)
    results=[]
    
    for _,row in recommendations.iterrows():
        directors_split=row['director'].split(',')
        
        if len(directors_split) == 2:
            direc='Directors'
            directors_str=f'{directors_split[0]} and {directors_split[1]}'
        elif len(directors_split) > 2:
            direc='Directors'
            directors_str=f'{", ".join(directors_split[:-1])} and {directors_split[-1]}'
        else:
            direc='Director'
            directors_str=row['director']

        year=int(row['date'])
        title=row['name']
        
        caption=f'{title} ({year})  {direc}: {directors_str}'       
        results.append((row['poster'],caption))
    return results

genres= ['All'] + sorted(movies['genre'].dropna().str.split(',').explode().str.strip().unique())
tones = ['All'] + ['Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad', 'Gross']

custom_css = """
    .centered-row {
        display: flex;
        align-items: center;
    }
    .button{
    background-color: blue;
    border-radius: 10px;
    }
    .gallery-container .grid-container.svelte-7anmrz.pt-6 {
        --grid-cols: 5 !important;
    }
    footer.svelte-zxu34v::before {
        content: "By Silon ·";
        margin-right: 7px;
    }
    """

with gr.Blocks(title='Movie Recommender') as dashboard:
    dashboard.theme=gr.themes.Monochrome() 
    gr.Markdown('# Semantic Movie Recommender')

    with gr.Row(elem_classes=['centered-row']):
        user_query=gr.Textbox(label='Enter a description of a movie', placeholder='E.g. shy girl falls in love', scale=1)
        genre_dropdown=gr.Dropdown(choices=genres, label='Select a genre', value='All', scale=1)
        tone_dropdown=gr.Dropdown(choices=tones, label='Select an emotional tone', value='All', scale=1)
        submit_button=gr.Button('Find recommendations', size='lg', scale=0, min_width=150, elem_classes=['button'])

    gr.Markdown("""
                <i>Note: If you are searching for a movie by its name | actors | director | year, you have to query: </i><b>inception | margot robbie | david fincher | 2023</b>
                """)

    gr.Markdown('## Recommendations:')
    output=gr.Gallery(label='Recommended',columns=5,object_fit='contain',height='auto')

    submit_button.click(fn=recommend_movies, inputs=[user_query,genre_dropdown,tone_dropdown], outputs=output)
    
if __name__ == '__main__':
    dashboard.launch(css=custom_css)