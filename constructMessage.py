def constructMessage(writing, shape, image, model, finetuning=False, solution=None):
    prompts = { "0": f"Which quadrant is the {shape} in this image located? In the case where the {shape} overlaps multiple quadrants, please provide the quadrant where the centre of the {shape} is located. Please only respond with 'Quadrant i' where i is the quadrant you think the {shape} is in. 1 is the top left, 2 is the top right, 3 is the bottom left, 4 is the bottom right. Do not reply with anything else.",
                "1": f"What are the coordinates of the {shape} in this image? The image is 400x400 pixels large, and the origin (0,0) is in the top left of the image. Please give your best estimate and do not reply with anything other than a set of coordinates.",
                "2": f"In the following image, one of the objects is different from the rest. Which quadrant of the image is the different object located? In the case where the object overlaps multiple quadrants, please provide the quadrant where the centre of the object is located. Please only respond with 'Quadrant i' where i is the quadrant you think the different object is in. 1 is the top left, 2 is the top right, 3 is the bottom left, 4 is the bottom right. Do not reply with anything else.",
                "3": f"In the following image, one of the objects is different from the rest. Please identify the coordinates of the different object. The image is 400x400 pixels large, and the origin (0,0) is in the top left of the image. Please give your best estimate and do not reply with anything other than a set of coordinates.", 
                "4": f"The image is divided into a 3x3 grid. Each element of the grid is referred to as a cell. Which cell is the {shape} in this image located? In the case where the {shape} overlaps multiple cells, please provide the cell where the centre of the {shape} is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",
                "5": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. Which cell is the {shape} in this image located? In the case where the {shape} overlaps multiple cells, please provide the cell where the centre of the {shape} is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",
               "std2x2": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. One of the objects in the image is the odd one out. In which cell is the odd object out in this image?  In the case where the objet overlaps multiple cells, please provide the cell where the centre of the object is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",
               "look2x2": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. One of the objects in the image is the odd one out. In which cell is the odd object out in this image?  In the case where the objet overlaps multiple cells, please provide the cell where the centre of the object is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else. Make sure you carefully consider each cell before submitting your response.",
               "pres OOO": f"In this image there are a number of objects. They are all identical except, possibly, for one. Can you identify whether the unique object is in the image? Please only answer with '1' if the object is present, and '0' if the object isn't present. Do not answer with anything else.",
               "std2x2-2Among5": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. In the presented image there are a number of objects. Almost all of the objects are the number 5 written as a numeral. There is a single 2 in the image, similarly represented by a numeral. In which cell is the 2 in? In the case where the 2 overlaps multiple cells, please provide the cell where the centre of the 2 is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",         
               "std2x2-5Among2": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. In the presented image there are a number of objects. Almost all of the objects are the number 2 written as a numeral. There is a single 5 in the image, similarly represented by a numeral. In which cell is the 5 in? In the case where the 5 overlaps multiple cells, please provide the cell where the centre of the 5 is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",              
               "presence-2Among5": f"In this image is a number of objects. Almost all of them are the numeral 5. There may or may not be a single numeral 2 in the image. Please only answer with '1' if the numeral '2' is present, and '0' if it isn't present. You may optionally add an explanation to qualify your answer if you are uncertain but please ensure you provide the boolean response first.",
               "coords-uncertainty": f"What are the coordinates of the {shape} in this image? The image is 400x400 pixels large, and the origin (0,0) is in the top left of the image. Please give your best estimate. You may qualify your answer with an explanation but please give the coordinates in brackets first.",
               "OOO-Uncertainty": f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. One of the objects in the image is the odd one out. In which cell is the odd object out in this image?  In the case where the objet overlaps multiple cells, please provide the cell where the centre of the object is located. Please respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). If you are uncertain please guess but optionally add a description to note this. However, for ease of processing please begin your response with 'Cell (i,j)'.",
               "std2x2-2Among5-conj": f"THe image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. In the presented image there are a number there are a number of objects. There are '2's and '5's written as numerals. In which cell is the green '2'? In the case where the green 2 overlaps multiple cells, please provide the cell where the centre of the 2 is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). If you are uncertain you may optionally add a note explaining that but please start your response with 'Cell (i,j)'."
              }
    message = []

    if model == "gpt-4o" or "llama" in model:
        message.append({"role": "system", "content": "You are an AI assistant that can analyze images and answer questions about them."})





    content = []
    
    if model =="llamaLocal":
        content.append({"type": "image"})


    content.append({"type": "text", "text": prompts[writing]})


    if model =="gpt-4o" or model=="llama11B" or model=="llama90B":
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})
    elif model == "claude-sonnet":
        content.append({"type": "image", "source": {"type": "base64", "media_type":"image/png", "data":image}})
    elif model == "llamaLocal":
        pass
    else:
        raise ValueError("Incorrect Model Type")

    message.append({"role": "user", "content":content})

    if finetuning:
        message.append({"role": "assistant", "content": str(solution)})



    return message
