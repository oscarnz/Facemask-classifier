from re import X
from tkinter import W
import streamlit as st
from PIL import ImageDraw
from server import load_image, process
import ast

#interact with FastAPI endpoint
url = "http://host.docker.internal:8000/"
endpoint = "api/prediction"

menu = ["Face Mask detection","PPE detection","Mysejahtera scanning"]
choice = st.sidebar.selectbox("Available applications",menu)

st.title('Face Mask detection')

uploaded_image = st.file_uploader("Upload Images", type=["jpg"])

if uploaded_image is not None:
    #feed uploaded image into var image
    image = load_image(uploaded_image)

    #pass uploaded image and api endpoint url into process function to perform post request from backend
    output = process(uploaded_image, url+endpoint)

    #function process will return json data object in string 
    #ast.literal will convert string into dictionary which is a valid datatype for json
    o = ast.literal_eval(output)

    #variable count for each label
    maskCount = 0
    nomaskCount = 0

    for i in range(len(o)):
        x = int(o[i]["x"])
        y = int(o[i]["y"])
        w = int(o[i]["w"])
        h = int(o[i]["h"])
        label = o[i]["label"]

        #create draw object for ImageDraw class
        draw = ImageDraw.Draw(image)

        #draw a rectangle based on the coordinates extracted by the haar carscade classifier
        if label == "Mask":
            draw.rectangle([x, y, x+h, y+w], fill=None, outline="green", width = 3)
            draw.text([x, y-10], "Mask", fill="green", stroke_width=10 )
            maskCount = maskCount+1
        else:
            draw.rectangle([x, y, x+h, y+w], fill=None, outline="red", width = 3)
            draw.text([x, y-10], "No Mask", fill="red", stroke_width=10 )
            nomaskCount = nomaskCount+1

    #get the input image type, size, dimension and filename to output
    file_details = {"filename":uploaded_image.name,
                    "filetype":uploaded_image.type,
                    "filesize":uploaded_image.size,
                    "dimension": str(image.width) + "x" + str(image.height)}
    st.write(file_details)
    st.image(image,width=None)

    #analysis of the prediction which consist of total of faces, number of faces with mask and without mask 
    st.header("Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected", maskCount+nomaskCount)
    col2.metric("With masks", maskCount)
    col3.metric("Without masks", nomaskCount)