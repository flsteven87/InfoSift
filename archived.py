    # wait for retired
    def generate_summary(self, file_name, transcript, max_token):
        llm = OpenAI(openai_api_key=openai.api_key, max_tokens=max_token)
        char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=30)
        transcripts = char_text_splitter.split_text(transcript)
        print(len(transcripts))
        docs = [Document(page_content=t) for t in transcripts]

        # Refine
        # prompt = '''
        # Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points by Tranditional Chinese.

        # {text}
        
        # And do it with Tranditional Chinese.
        # '''
        # # prompt_template = """Write a concise summary of the following: {text} CONCISE SUMMARY IN ITALIAN:"""
        # PROMPT = PromptTemplate(template=prompt, input_variables=["text"])
        # refine_template = (
        #     "Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points by Tranditional Chinese\n"
        #     "We have provided an existing summary newsletter up to a certain point: {existing_answer}\n"
        #     "We have the opportunity to refine the existing summary"
        #     "(only if needed) with some more context below.\n"
        #     "------------\n"
        #     "{text}\n"
        #     "------------\n"
        #     "Given the new context, refine the original newsletter."
        # )
        # refine_prompt = PromptTemplate(
        #     input_variables=["existing_answer", "text"],
        #     template=refine_template,
        # )
        # chain = load_summarize_chain(llm=llm, chain_type="refine", question_prompt=PROMPT, refine_prompt=refine_prompt)
        
        # Map-Reduce
        map_prompt = """
        Your job is to produce a map part summary of a map-reduce structure youtube transcript summarization.
        Write a map part summary of the following: \n
        {text}
        """
        combine_prompt_template = """
        Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points
        "{text}\n"
        """
        map_prompt_template = """Your job is to produce a map part summary of a map-reduce structure youtube video summarization.
        Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=map_prompt, combine_prompt=combine_prompt)
        summary = chain({"input_documents": docs}, return_only_outputs=True)

        with open(f"./summary/{file_name}.json", 'w') as f:
            json.dump(summary, f)

        with open(f"./summary/{file_name}.json") as f:
            data = json.load(f)
            summary = data['output_text']

        # response = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=prompt + transcript,
        #     max_tokens=max_token,
        #     n=1,
        #     stop=None,
        #     temperature=0.7,
        # )
        # with open(f"./transcript/{file_name}_summary.txt", "w") as text_file:
        #     text_file.write( response.choices[0].text)

        # return response.choices[0].text
        return summary

    # A function to transcript the audio file to text
    def speech_to_text(self, file_name):

        try:
            with open(f"./transcript/{file_name}.wav.txt") as file:
                transcript = file.read()
        except FileNotFoundError:
            print("File does not exist")
            print("Now calling API...")
            audio_file= open(f'./video/{file_name}.mp4', "rb")

            # 使用 OPENAI API
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            
            with open(f"./transcript/{file_name}.json", 'w') as f:
                json.dump(transcript, f)

            with open(f"./transcript/{file_name}.json") as f:
                data = json.load(f)
                transcript = data['text']

        return transcript