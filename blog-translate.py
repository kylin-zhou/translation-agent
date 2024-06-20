import os
import json

from icecream import ic
from langchain_community.document_loaders import FireCrawlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

import src.translation_agent as ta

def get_source_text_langchain(url):
    """
    scrape: Scrape single url and return the markdown.
    crawl: Crawl the url and all accessible sub pages and return the markdown for each one.
    """
    loader = FireCrawlLoader(
        api_key=key, url=url, mode="scrape"
    )

    docs = loader.load()
    return docs[0].page_content

def get_source_text(payload_url):
    import requests

    url = 'https://md.dhr.wtf/'
    params = {
        'url': payload_url
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        print(response.text)
    else:
        print(f'请求失败,状态码: {response.status_code}')
    return response.text

def split_chunks(source_text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(source_text)

    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    splits = [split.page_content for split in splits]
    
    return splits

def translate(source_lang, target_lang, source_text_chunks, country) -> str:
    """translate: Translate using various backends.

    Args:
        source_lang (string): Origin lang
        target_lang (string): The lang to translate the source to
        source_text (string): The text you want to translate.
        country (string): The country for google

    Returns:
        str: The translated text
    """

    ic("Translating text as multiple chunks")

    translation_2_chunks = ta.multichunk_translation(
        source_lang, target_lang, source_text_chunks, country
    )

    return "".join(translation_2_chunks)


if __name__ == "__main__":
    source_lang, target_lang, country = "English", "Chinese", "China"

    print(f"source text")
    url = "https://www.baseten.co/blog/llm-transformer-inference-guide/"
    source_text = get_source_text(url)

    print(f"Translation")
    source_text_chunks = split_chunks(source_text)
    translation = translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text_chunks=source_text_chunks,
        country=country,
    )

    # save source text and translation to json
    result = {"source": source_text, "translation": translation}
    with open(f"translation.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
