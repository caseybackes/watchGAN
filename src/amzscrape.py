from bs4 import BeautifulSoup
import os


class AmScrape():
    def __init__(self):
        self.dir = os.path.abspath('../data/webpages')
        self.__get_image_paths()

    def file2html(self, FILENAME):
        '''given a filename, return the BS4 object for the html file'''
        with open(os.path.join(self.dir, FILENAME), 'rb') as f:
            html = BeautifulSoup(f, 'html.parser')
            f.close()
        return html

    def __get_image_paths(self):
        dir_accumulator = []
        html_accumulator = []

        for directory in os.listdir(self.dir):
            
            fully_qualified_path = os.path.join(self.dir, directory)
            print('fully qualified path : ', fully_qualified_path)

            if os.path.isdir(fully_qualified_path):
                pass
            else:


        self.image_paths = acc
    


    
    def __repr__(self):
        return f"<AmScrape {self.dir}"


# Put all html files into a dictionary with key = filename reference and value = BS4-html object
if __name__ == "__main__":
    
    a = AmScrape()

    print('a.dir : ', a.dir)

    print('a.image_paths: ', a.image_pathss)


    
