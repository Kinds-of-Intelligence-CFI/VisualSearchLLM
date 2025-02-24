def constructMessage(writing, shape, image, model):

    prompts = [ f"Which quadrant is the {shape} in this image located? In the case where the {shape} overlaps multiple quadrants, please provide the quadrant where the centre of the {shape} is located. Please only respond with 'Quadrant i' where i is the quadrant you think the {shape} is in. 1 is the top left, 2 is the top right, 3 is the bottom left, 4 is the bottom right. Do not reply with anything else.",
                f"What are the coordinates of the {shape} in this image? The image is 400x400 pixels large, and the origin (0,0) is in the top left of the image. Please give your best estimate and do not reply with anything other than a set of coordinates.",
                f"In the following image, one of the objects is different from the rest. Which quadrant of the image is the different object located? In the case where the object overlaps multiple quadrants, please provide the quadrant where the centre of the object is located. Please only respond with 'Quadrant i' where i is the quadrant you think the different object is in. 1 is the top left, 2 is the top right, 3 is the bottom left, 4 is the bottom right. Do not reply with anything else.",
                f"In the following image, one of the objects is different from the rest. Please identify the coordinates of the different object. The image is 400x400 pixels large, and the origin (0,0) is in the top left of the image. Please give your best estimate and do not reply with anything other than a set of coordinates.", 
                f"The image is divided into a 3x3 grid. Each element of the grid is referred to as a cell. Which cell is the {shape} in this image located? In the case where the {shape} overlaps multiple cells, please provide the cell where the centre of the {shape} is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",
                f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. Which cell is the {shape} in this image located? In the case where the {shape} overlaps multiple cells, please provide the cell where the centre of the {shape} is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else.",
                f"The image is divided into a 2x2 grid. Each element of the grid is referred to as a cell. One of the objects in the image is the odd one out? In which cell is the odd object out in this image?  In the case where the objet overlaps multiple cells, please provide the cell where the centre of the object is located. Please only respond with 'Cell (i,j)' where (i,j) corresponds to the ith row and jth column of the grid. The top left cell is Cell (1,1). Do not reply with anything else."
                ]


    message = []

    if model == "gpt-4o":
        message.append({"role": "system", "content": "You are an AI assistant that can analyze images and answer questions about them."})





    content = []
    content.append({"type": "text", "text": prompts[writing]})


    if model =="gpt-4o":
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})
    elif model == "claude-sonnet":
        content.append({"type": "image", "source": {"type": "base64", "media_type":"image/png", "data":image}})
    elif model == "llama":
        content.append({"type": "image"})
    else:
        raise ValueError("Incorrect Model Type")

    message.append({"role": "user", "content":content})

    return message