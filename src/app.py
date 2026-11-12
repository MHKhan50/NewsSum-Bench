
import preprocessing
import streamlit as slt
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import nltk
import sentencepiece
path = ".\Colab models"
slt.set_page_config(layout="wide")

if slt.sidebar.checkbox("Upload File"):
     slt.title("Summarization")
     slt.sidebar.header("BBC News Articles Classification") 
     option = slt.sidebar.selectbox(
          'Select a model',
        ( 'Bart','Bert','MT5','T5','Pegasus','ALL_Models'))    

     
     uploaded_file = slt.file_uploader("Choose a File")
     if uploaded_file is not None :
        lines=[]
        if uploaded_file:
            for line in uploaded_file:
                data = line.decode("utf-8")
                lines.append(data)
            
                text  =' '.join([str(elem) for elem in lines])
   
     
   
#/Colab models/Bart/Bart_tokenizer


     if option == 'Bart' or option == 'Bert' or option == 'MT5' or option == 'T5' or option == 'Pegasus':

          if slt.button("Summarize"):
               with slt.spinner("Umair.."):
                    #file_tokenizer = "\Colab models\{}\{}_tokenizer".format(option,option)
                    #file_model = "\Colab models\{}\{}_model".format(option,option)
                    #tokenizer = AutoTokenizer.from_pretrained(file_tokenizer)
                    #model = AutoModelForSeq2SeqLM.from_pretrained(file_model)
                    model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)
                    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    summary = preprocessing.generate_summary(text,tokenizer,model)
                    slt.success("Summary Generated")
                    slt.header(option)
                    slt.write(summary)
                    slt.download_button('Download Summary', summary) 

          
    
     if option == 'ALL_Models' :
          if slt.button("Summarize"):
               with slt.spinner("Umair.."):
                    lis = ['Bart','Bert','MT5','T5','Pegasus']
                    for i in lis:
                         file_tokenizer_new = "\Colab models\{}\{}_tokenizer".format(i,i) 
                         file_model_new = "\Colab models\{}\{}_model".format(i,i)
                         tokenizer_new = AutoTokenizer.from_pretrained(file_tokenizer_new)
                         model_new = AutoModelForSeq2SeqLM.from_pretrained(file_model_new)
                         summary =preprocessing.generate_summary(text,tokenizer_new,model_new)
                         slt.header(i)
                         slt.write(summary)
                         slt.download_button('Download Summary', summary)

elif slt.sidebar.checkbox("Only_Text"):

     
     
     slt.title("Summarization")
     slt.sidebar.header("BBC News Articles Classification") 
     text = slt.text_area("BBC Articles")
     option = slt.sidebar.selectbox(
          'Select a model',
        ( 'Bart','Bert','MT5','T5','Pegasus','ALL_Models'))    



     if option == 'Bart' or option == 'Bert' or option == 'MT5' or option == 'T5' or option == 'Pegasus':

          if slt.button("Summarize"):
               with slt.spinner("Umair.."):
                    file_tokenizer = "\Colab models\{}\{}_tokenizer".format(option,option)
                    file_model = "\Colab models\{}\{}_model".format(option,option)     
                    tokenizer = AutoTokenizer.from_pretrained(file_tokenizer)
                    model = AutoModelForSeq2SeqLM.from_pretrained(file_model)
                    summary = preprocessing.generate_summary(text,tokenizer,model)
                    slt.success("Summary Generated")
                    slt.header(option)
                    slt.write(summary)
                    slt.download_button('Download Summary', summary) 

          
    
     if option == 'ALL_Models' :
          if slt.button("Summarize"):
               with slt.spinner("Umair.."):
                    lis = ['Bart','Bert','MT5','T5','Pegasus']
                    for i in lis:
                         file_tokenizer_new = "\Colab models\{}\{}_tokenizer".format(i,i) 
                         file_model_new = "\Colab models\{}\{}_model".format(i,i)
                         tokenizer_new = AutoTokenizer.from_pretrained(file_tokenizer_new)
                         model_new = AutoModelForSeq2SeqLM.from_pretrained(file_model_new)
                         summary =preprocessing.generate_summary(text,tokenizer_new,model_new)
                         slt.header(i)
                         slt.write(summary)
                         slt.download_button('Download Summary', summary)



           
            



        
        




