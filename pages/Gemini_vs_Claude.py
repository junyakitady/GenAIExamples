import os
import tempfile
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from anthropic import AnthropicVertex, AsyncAnthropicVertex
import asyncio
from pypdf import PdfReader

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "asia-northeast1"
vertexai.init(project=PROJECT_ID, location=REGION)

# layoout page
st.set_page_config(layout="wide")
st.title("Gemini vs Claude3")
with st.form('my_form'):
    uploaded_file = st.file_uploader("(option) PDF を入力する場合、Gemini は画像として、Claude はテキストとしてロードします。", type=["pdf"],)
    if uploaded_file:
        with st.spinner('Loading ...'):
            bytes_data = uploaded_file.getvalue()
            # for Gemini
            document1 = Part.from_data(data=bytes_data, mime_type="application/pdf")
            # for Anthropic
            with tempfile.TemporaryDirectory() as temp_dir:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as temp_pdf:
                    temp_pdf.write(bytes_data)
                    reader = PdfReader(temp_pdf.name)
                    pdf_text = ''.join([page.extract_text() for page in reader.pages])

    input = st.text_area(label='Prompt', value='映画好きの人にお勧めの日本国内の名所を教えてください')
    submitted = st.form_submit_button('Submit')

col1, col2 = st.columns(2)
col1.subheader('Gemini 1.5 Pro')
col2.subheader('Claude 3 Opus')


async def ask_gemini15pro():
    if uploaded_file:
        data = [document1, input]
    else:
        data = [input]
    with col1:
        fulltext = ""
        message_placeholder = st.empty()
        model = GenerativeModel("gemini-1.5-pro")
        responses = await model.generate_content_async(data, generation_config={"max_output_tokens": 2048, "temperature": 1.0}, stream=True)
        async for response in responses:
            fulltext += response.text
            message_placeholder.markdown(fulltext)


async def ask_claude():
    if uploaded_file:
        data = "Context:\n" + pdf_text + "\n=======\n" + input
    else:
        data = input
    with col2:
        fulltext = ""
        message_placeholder = st.empty()
        client = AsyncAnthropicVertex(region="us-east5", project_id=PROJECT_ID)
        async with client.messages.stream(model="claude-3-opus@20240229", max_tokens=2048, messages=[{"role": "user", "content": data, }]) as stream:
            async for text in stream.text_stream:
                fulltext += text
                message_placeholder.markdown(fulltext)


async def ask_llms():
    tasks = []
    tasks.append(asyncio.create_task(ask_gemini15pro()))
    tasks.append(asyncio.create_task(ask_claude()))
    await asyncio.gather(*tasks)


if submitted:
    asyncio.run(ask_llms())
