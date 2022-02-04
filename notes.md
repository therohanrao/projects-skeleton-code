# Done on 2/32/32/3
1. look at pillow -> convert image to pytorch tensor
2. get data about size of dataset
3. csv file -> get images? (pass in csv file)
* Pandas: path to csv file, load csv into pandas dataframe
* basically a big array of images, easier to work with


# 1. Pillow - Used to manipulate images
>from PIL import Image
>import torchvision.transforms as transforms

Won't work since I don't have a file called 'cat.png'
>image = Image.open('cat.png')
>image_tensor = transforms.ToTensor()(image)

# 2. Pandas - Used to work with data

How you will probably use pandas
>import pandas as pd
>pd.read_csv('somefile.csv')

DataFrame's are just tables
data = {
    "a": [1, 2, 3],
    "b": [23, 23, 23]
}

# Example dataframe acess
>df = pd.DataFrame(data)
>df.loc[row, col]
you can access cols by their string names (at top of csv)

example debug code
> print(self.images.info())
> print(len(self.images))
