from typing import List

import chainlit as cl
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.retrievers import TavilySearchAPIRetriever
from tqdm import tqdm

model_name = 'qwen:7b'
model = Ollama(base_url='http://localhost:11434', model=model_name)
text_splitter = RecursiveCharacterTextSplitter(
    ["\n\n\n", "\n\n", "\n"],
    chunk_size=400,
    chunk_overlap=50
)
embedding_model = OllamaEmbeddings(
    base_url='http://localhost:11434',
    # model='nomic-embed-text'
    model='znbang/bge:large-zh-v1.5-q8_0'
)


def search_with_ddg(query):
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        search_results = [r for r in ddgs.text(
            query,
            # region='zh-cn',
            max_results=10)]
        # 规范命名
        for item in search_results:
            item['abstract'] = item['body']
            del item['body']
    return search_results


def search_with_tavily(query, max_retry=3):
    # os.environ["TAVILY_API_KEY"] = '替换为自己的'
    retriever = TavilySearchAPIRetriever(k=6)
    documents = retriever.invoke(query)
    return [{'title': doc.metadata['title'], 'abstract': doc.page_content, 'href': doc.metadata['source'], 'score': doc.metadata['score']} for doc in
            documents]


def search_with_bing(query, k=5):
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import quote
    url = f'https://cn.bing.com/search?q={quote(query)}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    }
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # #b_results > li.b_ad.b_adTop > ul > li.b_adLastChild > div > h2
    result_elements = soup.select('#b_results > li')
    data = []
    # link: div.b_tpcn > a > div.tptxt > div.tpmeta > div > cite
    # abstract: div.b_caption > p
    # #b_results: h2
    for parent in result_elements:
        if parent.select_one('h2') is None:
            continue
        data.append({
            'title': parent.select_one('h2').text,
            'abstract': parent.select_one('div.b_caption > p').text.replace('\u2002', ' '),
            'href': parent.select_one('div.b_tpcn > a').get('href')
        })
    return data[:k]


async def retrieve(query, k=5):
    async with cl.Step(name="搜索引擎检索") as step:
        step.input = query
        # search_results = search_with_ddg(query)
        # search_results = search_with_tavily(query)
        search_results = search_with_bing(query, k)
        step.output = search_results
    search_results = {item['title']: item for item in search_results}

    async with cl.Step(name="获取网页全文") as step:
        step.input = [{
            'url': item['href'],
            'title': item['title']
        } for item in search_results.values()]
        for item in tqdm(search_results.values(), desc='获取网页全文'):
            try:
                # 请求要要try，防止网页请求失败
                resp = requests.get(item['href'])
                # 部分网页不规范，Header中没有charset，先直接尝试gb18030，如果失败再尝试utf-8
                html = resp.content.decode('gb18030')
            except Exception as e:
                print(f"{item['href']}, {e}")
                try:
                    html = resp.content.decode('utf-8')
                except Exception as e:
                    print(f"原文获取失败{item['href']}, {e}")
                    html = ''

            soup = BeautifulSoup(html, 'html.parser')
            item['body'] = soup.get_text()
            if item['body'] is not None:
                item['body'] = item['body'].strip()
            else:
                item['body'] = ''

            if item['body'].strip() == '' or len(item['body']) < len(item['abstract']):
                item['body'] = item['abstract']
        step.output = [{
            'url': item['href'],
            'title': item['title'],
            'abstract': item['abstract'],
            'body': item['body']
        } for item in search_results.values()]

    async with cl.Step(name="向量检索") as step:
        documents = [Document(
            item['body'],
            metadata={'href': item['href'], 'title': item['title']}
        ) for item in search_results.values()]
        split_docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(split_docs, embedding_model)
        retriever = vectorstore.as_retriever(search_args={'k': k})
        retrieved_docs = retriever.get_relevant_documents(query)
        context = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        step.output = '\n\n'.join([f"参考片段{idx + 1}\n标题：{doc.metadata['title']}\n链接：{doc.metadata['href']}\n片段：{doc.page_content}" for idx, doc in
                                   enumerate(retrieved_docs)])
    return search_results, retrieved_docs, context


async def llm_generator(prompt):
    for chunk in model.stream(prompt):
        # print(chunk, end='')
        yield chunk


@cl.on_message
async def main(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler()

    query = message.content
    search_results_dict, retrieved_docs, context = await retrieve(query)
    if len(search_results_dict) == 0:
        raise ValueError('搜索结果为空')

    prompt = """请使用下方的上下文（<<<context>>><<</context>>>之间的部分）回答用户问题，如果所提供的上下文不足以回答问题，请回答“我无法回答这个问题”
<<<context>>>
{context}
<<</context>>>

用户提问：{query}
请回答：
    """.format(query=query, context=context)

    text_elements = []  # type: List[cl.Text]
    source_names = set()
    for source_idx, source_doc in enumerate(retrieved_docs):
        # Create the text element referenced in the message
        title = source_doc.metadata['title']
        print(title)

        if title in source_names:
            continue

        if title not in search_results_dict:
            print('=' * 50)
            print(f'title not found, title: {title}, search_result_dict.keys(): {search_results_dict.keys()}')
            print('=' * 50)
            continue

        search_result = search_results_dict[title]
        preview = search_result['abstract']
        if len(preview) > 150:
            preview = preview[:150] + '...'
        text_elements.append(
            cl.Text(content=f"{search_result['href']}\n{preview}", name=title)
        )
        source_names.add(title)

    msg = cl.Message(content="")
    answer = ''
    async for chunk in llm_generator(prompt):
        answer += chunk
        await msg.stream_token(chunk)

    await cl.Message(content='来源：\n', elements=text_elements).send()
