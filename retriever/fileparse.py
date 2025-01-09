import os
import queue
import io
from spire.doc import *
from spire.doc.common import *
from spire.pdf import PdfDocument,PdfTextExtractOptions,PdfTextExtractor
from spire.pdf import *
from spire.pdf.common import *
from PIL import Image
from typing import Iterable
from src.modules.extras.tagger.core.interrogator import Interrogator
from src.modules.extras.tagger.core.interrogators import interrogators

class WordExtractor:

    def __init__(self, file_Location: str = None, workspace: str = None) -> None:
        self.file_location = file_Location
        self.workspace = workspace if workspace else os.path.dirname(file_Location)
        os.makedirs(self.workspace, exist_ok=True)
        print("[WordExtractor] 工作空间：", self.workspace)

    def extract_text(
        self, return_value: bool = True, save_output: bool = False
    ) -> (tuple[str, str] | str):
        """
        读取word文档中的文本和表格内容，并将其保存为txt文件，文件名为_text.txt，保存在output_folder_path目录下，返回文件路径
        """

        def process_image(element, index):
            picture = element
            data_bytes = picture.ImageBytes
            image = Image.open(io.BytesIO(data_bytes))
            image_output_path = os.path.join(self.workspace, "imgs", f"image_{index}.png")
            os.makedirs(os.path.join(self.workspace, "imgs"), exist_ok=True)
            image.save(image_output_path)
            return "\n图片内容：" + ImageReader().generate_tags(image_output_path) + "\n"

        document = Document()
        document.LoadFromFile(self.file_location)
        document_text = ""
        images_text = []  # 用于存储提取的图片文本
        image_index = 1

        for s in range(document.Sections.Count):
            section = document.Sections.get_Item(s)
            body_elements = []

            # 收集段落和表格
            for i in range(section.Body.ChildObjects.Count):
                element = section.Body.ChildObjects.get_Item(i)
                body_elements.append(element)

            for element in body_elements:
                if isinstance(element, Paragraph):
                    # 提取段落文本
                    print("[WordExtractor] 提取段落文本......")

                    for i in range(element.ChildObjects.Count):
                        subobj = element.ChildObjects[i]
                        #print("子元素类型：", type(subobj))
                        if isinstance(subobj, DocPicture):
                            document_text += process_image(subobj, image_index)
                            image_index += 1
                        else:
                            if hasattr(subobj, 'Text'):
                                document_text += subobj.Text + "\n"
                elif isinstance(element, Table):
                    # 提取表格文本
                    print("[WordExtractor] 提取表格文本......")
                    for r in range(element.Rows.Count):
                        row = element.Rows.get_Item(r)
                        rowData = []
                        for c in range(row.Cells.Count):
                            cell = row.Cells.get_Item(c)
                            cellText = ''
                            for para in range(cell.Paragraphs.Count):
                                paragraphText = cell.Paragraphs.get_Item(para).Text
                                cellText += paragraphText.strip() + ' '
                            rowData.append(cellText.strip())
                        document_text += '\t'.join(rowData) + '\n'
                elif isinstance(element, DocPicture):
                    # 提取图片文本
                    print("[WordExtractor] 提取图片文本......")
                    document_text += process_image(element, image_index)
                    image_index += 1
        document_text = document_text.replace("\n：", "：")
        origin_filename = os.path.basename(self.file_location)
        output_file_path = os.path.join(self.workspace, f"{origin_filename}_text.txt")

        if save_output:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(document_text)
            print("[WordExtractor] 已提取文档内容并保存到文件：", output_file_path)
            if return_value:
                return output_file_path, document_text
            else:
                return output_file_path
        else:
            return document_text

    def extract_table(
        self, return_value: bool = True, save_output: bool = True
    ) -> (tuple[str, str] | str):
        """
        读取word文档中的表格内容，并将其保存为txt文件，文件名为_table.txt，保存在output_folder_path目录下，返回文件路径
        """
        document = Document()
        input_file = self.file_location
        # 加载word文档
        document.LoadFromFile(self.file_location)
        tableData = ''
        for s in range(document.Sections.Count):
            section = document.Sections.get_Item(s)
            tables = section.Tables
            for i in range(0,tables.Count):
                # 获取一个表格
                table = tables.get_Item(i)
                for j in range(0,table.Rows.Count):
                    # 遍历行的单元格
                    rowData = []
                    for k in range(0,table.Rows.get_Item(j).Cells.Count):
                        # 获取一个单元格
                        cell = table.Rows.get_Item(j).Cells.get_Item(k)
                        cellText = ''
                        for para in range(cell.Paragraphs.Count):
                            paragraphText = cell.Paragraphs.get_Item(para).Text
                            cellText += (paragraphText+' ')
                        rowData.append(cellText.strip())
                    tableData += '\t'.join(rowData) + '\n'
        document.Close()            
        # 将表格数据保存在txt文件
        origin_filename = os.path.basename(input_file)
        output_file_path = os.path.join(self.workspace, f"{origin_filename}_table.txt")

        if save_output:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(tableData)
            print("[WordExtractor] 已提取文档内容并保存到文件：", output_file_path)

            if return_value:
                return output_file_path, tableData
            else:
                return output_file_path
        else:
            return tableData

    def extract_image(
        self, return_value: bool = True, save_output: bool = False
    ) -> (tuple[str, str] | str):
        """
        读取word文档中的图片，并将其保存为jpg文件，文件名为_picture.jpg，保存在output_folder_path目录下
        反推一个word文档中每张图片的提示词并写入txt文件，返回文件路径
        """
        input_file = self.file_location
        document = Document()
        document.LoadFromFile(input_file)
        # 创建一个列表储存提取的图片数量
        images = []
        # 初始化一个队列来存储待遍历的文档元素
        nodes = queue.Queue()
        nodes.put(document)

        # 遍历文档元素
        while not nodes.empty():
            node = nodes.get()
            for i in range(node.ChildObjects.Count):
                obj = node.ChildObjects[i]
                # 查找图片
                if isinstance(obj, DocPicture):
                    picture = obj
                    # 将图片数据添加到列表中
                    data_bytes = picture.ImageBytes
                    images.append(data_bytes)
                elif isinstance(obj, ICompositeObject):
                    nodes.put(obj)
        document.Close()
        # 反推提示词储存在txt文件
        origin_filename = os.path.basename(input_file)
        txtfile_output_path = os.path.join(self.workspace, f"{origin_filename}_prompt.txt")

        output_file_paths = []
        image_tags = []
        for i, image_data in enumerate(images):
            output_file_path = os.path.join(self.workspace, f"imgs/image_{i}.png")
            output_file_paths.append(output_file_path)
            os.makedirs(os.path.join(self.workspace, "imgs"), exist_ok=True)
            # 将字节数组转换为图像
            image = Image.open(io.BytesIO(image_data))
            # 保存图像
            image.save(output_file_path)
            print("[WordExtractor] 图片保存到文件：", output_file_path)
            image_tags.append(ImageReader().generate_tags(output_file_path))
            print("[WordExtractor] 图片反推成功")

        if save_output:
            with open(txtfile_output_path, "a", encoding="utf-8") as f:
                f.writelines(tag + "\n" for tag in image_tags)  # 使用生成器表达式
            print("[WordExtractor] 已反推图片提示词并保存到文件：", txtfile_output_path)
    
            if return_value:
                return txtfile_output_path, image_tags
            else:
                return txtfile_output_path
        else:
            return image_tags


    def operate(self, input, config):
        pass

class PdfExtractor:
    file_location = None
    def __init__(self, file_Location: str = None) -> None:
        self.file_location = file_Location
        

    def extract_text(self,output_folder_path:str)->str:
        """
        把pdf文件里所有文字性内容（包括表格内容）提取出来保存在txt文件并返回文件路径
        """
        pdf = PdfDocument()
        pdf.LoadFromFile(self.file_location)
        input_file = self.file_location
        extracted_text = ""
        extract_options = PdfTextExtractOptions()
        # 循环遍历文档中的页面
        for i in range(pdf.Pages.Count):
            # 获取页面
            page = pdf.Pages.get_Item(i)
            # 创建PdfTextExtractor对象，并将页面作为参数传递
            text_extractor = PdfTextExtractor(page)
            # 从页面中提取文本
            text = text_extractor.ExtractText(extract_options)
            # 将提取的文本添加到字符串对象中
            extracted_text += text

        last_part = os.path.basename(input_file)
        txtfile_output_path = os.path.join(output_folder_path,f"{last_part}_text.txt")
        # 将提取的文本写入文本文件
        with open(txtfile_output_path, "w",encoding="utf-8") as f:
            f.write(extracted_text)
        pdf.Close()
        return txtfile_output_path
    

    def extract_image(self,output_folder_path:str)->str:
        pdf = PdfDocument()
        pdf.LoadFromFile(self.file_location)
        input_file = self.file_location
        image_helper = PdfImageHelper()
        last_part = os.path.basename(input_file)
        image_count = 1
        prompt_file_path = os.path.join(output_folder_path,f"{last_part}_prompt.txt")
        for i in range(pdf.Pages.Count):
            images_info = image_helper.GetImagesInfo(pdf.Pages[i])
            # 保存图片并返回抽取的提示词文件
            for j in range(len(images_info)):
                image_info = images_info[j]
                output_file_path = os.path.join(output_folder_path,f"{last_part}_image{image_count}.png")
                image_info.Image.Save(output_file_path)
                image_count += 1
                prompt = ImageReader().generate_tags(output_file_path)
                with open(prompt_file_path,"w",encoding="utf-8") as f:
                    f.write(prompt)

        pdf.Close()
        return prompt_file_path
  
class ImageReader:
    
    def __init__(
        self, 
        model: str = "wd-vit-large-tagger-v3",
        threshold: float = 0.35,
        exclude_tags: list = None,
        rawtag : bool = False,
        platform: str = "CPU"
    ) -> None:
        
        self.model = model
        self.interrogator = interrogators[self.model]
        self.threshold = threshold
        self.exclude_tags = exclude_tags
        self.rawtag = rawtag

        if platform == "CPU":
            self.interrogator.use_cpu()
            print("[PictureReader] using CPU")

    def parse_exclude_tags(self) -> set[str]:
        if self.exclude_tags is None:
            return set()

        tags = []
        for str in self.exclude_tags:
            for tag in str.split(','):
                tags.append(tag.strip())

        # reverse escape (nai tag to danbooru tag)
        reverse_escaped_tags = []
        for tag in tags:
            tag = tag.replace(' ', '_').replace('\(', '(').replace('\)', ')')
            reverse_escaped_tags.append(tag)
        return set([*tags, *reverse_escaped_tags])  # reduce duplicates

    def image_interrogate(
        self, image_path: str, tag_escape: bool, exclude_tags: Iterable[str]
    ) -> dict[str, float]:
        """
        Predictions from a image path
        """
        im = Image.open(image_path)
        result = self.interrogator.interrogate(im)

        return Interrogator.postprocess_tags(
            result[1],
            threshold=self.threshold,
            escape_tag=tag_escape,
            replace_underscore=tag_escape,
            exclude_tags=exclude_tags)
    
    def generate_tags(self, image_path) -> str:
        tags = self.image_interrogate(image_path, not self.rawtag, self.parse_exclude_tags())
        tags_str = ', '.join(tags.keys())
        return tags_str
  



if __name__ == "__main__":
    # python -m src.modules.agent.retriever.fileproc
    f = "/home/legion4080/AIPJ/MYXY/assets/space crisis.docx"
    ext = WordExtractor(f, "./tmp")

    o = ext.extract_text(return_value=True, save_output=True)
    print(o)
    #i = "/home/legion4080/AIPJ/MYXY/assets/three/skybox/back.jpg"

    #p = PictureReader(file_Location=i)
    #print(p.extract_picture())
