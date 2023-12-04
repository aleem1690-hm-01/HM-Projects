import streamlit as st

import os
import langchain

from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper

import openai

from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import WebBaseLoader

import tiktoken

#Updating the server key for google search
# os.environ["SERPER_API_KEY"] = "b428c3270564754f69383922cd85763cb7b79533"
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

#Updating the Azure open Api Key & End Point for model
# os.environ["AZURE_OPENAI_API_KEY"] = 'f7e330db85eb4ca5855620ca2656871e'
# os.environ["AZURE_OPENAI_ENDPOINT"] = 'https://openaitrials.openai.azure.com/'


os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]

class PlayerSummary:
  def __init__(self,mvp_player_name,tags):
    model = AzureChatOpenAI(
                    azure_deployment='gpt35turbo16kdep2',
                    openai_api_version="2023-07-01-preview",
                    openai_api_type="azure",
                    )

    self._mvp = mvp_player_name
    self._tags = tags
    self._model = model

  def get_href_links(self):
    
    mvp_player_name = self._mvp
    tags = self._tags

    # print(mvp_player_name)
    search = GoogleSerperAPIWrapper(type="news",tbs="qdr:m6")
    search_term = mvp_player_name + " recent " + tags
    # st.write(search_term)
    results = search.results(search_term)
    
    href_list = []
    
    for links in results["news"]:
      href_list.append(links["link"])

    return href_list
  
  def num_tokens_from_string(self,string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

  def get_individual_summary(self):

    mvp_player_name = self._mvp
    tags = self._tags
    model = self._model

    href_list = self.get_href_links()
    loader = WebBaseLoader(href_list)
    docs = loader.load()

    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
        )
    
    document_variable_name = "context"
    prompt = PromptTemplate.from_template(
        "Summarize the article based on {mvp_player_name} alone: {context}. Ignore details of any other player"
        )
    
    llm_chain = LLMChain(llm=model, prompt=prompt)

    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name
        )
    
    summary_docs = ""

    for idx in range(len(href_list)):
      loader = WebBaseLoader(href_list[idx])
      count = str(idx+1)
      docs = loader.load()
      docs_dict = dict(docs[0])
      no_tokens = self.num_tokens_from_string(docs_dict["page_content"],"cl100k_base")
      if no_tokens<16000:
        summary = chain.run(mvp_player_name=mvp_player_name,input_documents=docs)
        summary_docs = summary_docs +'\nSummary Number '+count+':\n'+summary
        # print('article no:'+count+'\n'+summary)
      else:
        print("skipped article no:"+count+" due to tokens exceeding :"+str(no_tokens))

    return summary_docs

  def get_final_summary(self):
    
    mvp_player_name = self._mvp
    tags = self._tags
    model = self._model
    
    summary_docs = self.get_individual_summary()
    summary_template = """
    
    I am building a team in fantasy application and looking to select players based on

    Requirements:
    a) Recent performances
    b) form, skills, abilities
    c) injuries or availability issues that might impact selection

    Output Examples:

    "1) Xyz had a big third quarter that saw him record the best score for the game with 21 of his 32 touches being contested as he also had 10 tackles.
    2) Xyz had 23 touches of which 12 were contested and he had 8 tackles. Not a good score for him but the Dogs werent good which didnt help
    3) Xyz was brilliant this week with 32 touches (16 contested), 9 tackles & 3 goals. A solid C or VC option each week.

    I want you to 
    
    a) read the given document which contains summaries from multiple articles
    b) if the document is empty, return output as "No data available"
    b) Exclude all opinions and suggestions from other people. Focus only on facts.
    c) consolidate & provide an overall summary in the "Output Examples" format for {mvp_player_name} and above "Requirements".
    d) The summary should be specific to player's on-field numbers and scores.
    e) The summary should be very concise and brief.
    f) Give the summary as a paragraph and not as bullet points.
    g) dont mention details of any other player unless it impacts the selecton of {mvp_player_name} 
    
    Document: {doc_summaries}

    Output:"""

    summary_prompt = PromptTemplate.from_template(summary_template)
    # summary_prompt

    summary_chain = LLMChain(llm=model, prompt=summary_prompt)

    summary = summary_chain.run(doc_summaries=summary_docs,mvp_player_name=mvp_player_name)

    refined_summary_template = """In the summarized Document below, i want you to refine the summary further by:

    a) Being concise
    b) Focussing on on-field performances 
    c) Excluding generic information
    d) Excluding information that cant be quantified
    e) Excluding any conclusions
    f) Excluding any suggestion & remarks

    Document: {refined_summary}

    Output:    
    """
    refined_summary_prompt = PromptTemplate.from_template(refined_summary_template)

    refined_summary_chain = LLMChain(llm=model, prompt=refined_summary_prompt)

    final_summary = refined_summary_chain.run(refined_summary=summary)

    return final_summary

def main(mvp_player_name, tags):
  
  st.write(f"Player Name: {mvp_player_name}")
  st.write(f"Search Criteria: {tags}")
  # mvp_player_name = "Christian Petracca"
  # tags = "AFL scores performance"
  player = PlayerSummary(mvp_player_name,tags)
  summary = player.get_final_summary()
  # print("Player Summary is:"+'\n'+summary)
  st.write("Player Summary is:")
  st.write(summary)
  return summary

if __name__ == "__main__":
  # Set title of the page
  st.title("NewsCorp Player Summary")
  # Input for entering Player Name
  mvp_player_name = st.text_input("Enter Player Name")
  # Radio button for selecting League
  league_options = ['AFL','BBL','NBL','NRL']
  selected_league = st.radio("League", league_options)

  # Input for entering Search Tags
  search_tags = st.text_input("Enter Search Tags")

  # Concatenate player name and search tags
  #tags = f"League: {selected_league}, ags: {search_tags}"
  tags = selected_league+ " " + search_tags

  # mvp_player_name = "Christian Petracca"
  # tags = "AFL scores performance"
  # player = PlayerSummary(mvp_player_name,tags)
  # summary = player.get_final_summary()

  if st.button("Submit"):
    main(mvp_player_name, tags)

  # print('read summary from main function')
  # summary
