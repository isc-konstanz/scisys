import pandas as pd
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import *
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Paragraph
from PIL import Image
import datetime as dt
import toml
import os


# Creating canvas and setting global variables #####################################################################
class PDF:
    def __init__(self, project_dir, pagesize,
                 left_margin, right_margin, top_margin, bottom_margin,
                 font_content, font_bold, fontsize_content, fontsize_table):

        self.project_dir = project_dir

        with open(os.path.join(project_dir, "conf", "pdf_settings.cfg"), mode="rt") as config_file:
            config = toml.load(config_file)
            self.general_dict = config["General"]
            self.pictures_dict = config["Pictures"]
            self.pages_dict = config["Pages"]
            self.appendix_dict = config["Appendix"]

            self.table_of_content = []

            for page in config["Pages"]:
                self.table_of_content.append(config["Pages"][page].replace("_", " ").title())
            if len(self.appendix_dict) > 0:
                self.table_of_content.append("Appendix")

            self.table_of_content_appendix = []

            for page in config["Appendix"]:
                self.table_of_content_appendix.append(config["Appendix"][page].replace("_", " ").title())

        self.pagesize = pagesize
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.mid_page = (right_margin + left_margin) / 2
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.font_content = font_content
        self.font_bold = font_bold
        self.fontsize_content = fontsize_content
        self.fontsize_table = fontsize_table

        if self.general_dict["date"] == "today":
            self.general_dict["date"] = dt.date.today()
        else:
            self.general_dict["date"] = self.general_dict["date"]

        self.pdf_path = os.path.join(project_dir, str(self.general_dict["date"]) + " " +
                                     self.general_dict["title"] + " " +
                                     self.general_dict["project"] + ".pdf")

        self.canvas = Canvas(self.pdf_path, pagesize=pagesize)

        try:
            if os.path.isfile(self.pdf_path):
                os.remove(self.pdf_path)

        except PermissionError:
            self.pdf_path = os.path.join(project_dir, str(self.general_dict["date"]) + " " +
                                         self.general_dict["title"] + " " +
                                         self.general_dict["project"] + "(2).pdf")
            self.canvas = Canvas(self.pdf_path, pagesize=pagesize)

        print("[PDFCreator] Initiating PDF " + self.general_dict["title"] + " for Project "
              + self.general_dict["project"])

    def draw_layout(self, logo_dir: str, logo: str, headline: str = "", explanation: str = ""):
        """
        Draws a layout with logo, Header and Footer.

        :param logo_dir: Path to the logo file directory.
        :param logo: Name of the logo file.
        :param headline: Headline, written in the Header
        :param explanation: Sub-Headline, explaining the page content.
        :return:
        """

        print("[PDFCreator] Creating Page " + headline)
        # Header
        self.canvas.drawImage(os.path.join(logo_dir, logo),
                              self.left_margin * mm, 280 * mm, width=11 * mm, height=11 * mm)
        self.canvas.setFont(self.font_bold, 20)
        self.canvas.setFillColorRGB(0.1, 0.3, 0.5)
        self.canvas.drawCentredString(105 * mm, 282 * mm, headline)
        self.canvas.setFont(self.font_content, self.fontsize_content)
        self.canvas.drawCentredString(105 * mm, 275 * mm, explanation)
        self.canvas.setFillColor("black")
        self.canvas.setFont(self.font_content, 12)
        self.canvas.drawCentredString(190 * mm, 282 * mm, str(self.general_dict["confidentiality"]))

        # Footer
        self.canvas.setStrokeColorRGB(0.5, 0.5, 0.5)
        self.canvas.line(self.left_margin * mm, 15 * mm, self.right_margin * mm, 15 * mm)
        self.canvas.setStrokeColor("black")
        self.canvas.setFont(self.font_content, 12)
        self.canvas.drawCentredString(105 * mm, 7 * mm,
                                      self.general_dict["title"] + " " + self.general_dict["project"])
        self.canvas.drawString(190 * mm, 7 * mm, str(self.canvas.getPageNumber()))
        self.canvas.drawString(self.left_margin * mm, 7 * mm, str(self.general_dict["date"]))

    def insert_image(self, image_dir, image_file: str, x_cord: int, y_cord: int, width: int = None, height: int = None):
        """
        Inserts an image in the current PDF Page.

        :param image_dir: Path to the image file directory.
        :param image_file: Name of the image file.
        :param x_cord: x-coordinate the image will be inserted at in the current PDF Page
        :param y_cord: y-coordiante the image will be inserted at in the current PDF Page
        :param width: Maximum width of the image. Ratio will be kept.
        :param height: Maximum height of the image. Ratio will be kept.
        :return:
        """

        image = Image.open(os.path.join(self.project_dir, image_dir, image_file))

        img_height = image.height
        img_width = image.width

        if image.width > (self.right_margin - x_cord):
            img_width = (self.right_margin - x_cord)
            img_height = img_height / (image.width / img_width)

        if width is not None:
            img_width = width
            img_height = img_width / (image.width / image.height)

        if height is not None:
            img_height = height
            img_width = img_height / (image.height / image.width)

        self.canvas.drawImage(os.path.join(self.project_dir, image_dir, image_file), x_cord * mm, y_cord * mm,
                              width=img_width * mm, height=img_height * mm)

    def insert_textlines(self, text: str or list, x_cord: int, y_cord: int, textcolor: str = "black",
                         font: str = "", fontsize: int = 0):
        """
        Inserts textlines, either from a string or a list.

        :param text: Text string or list. Every String in a list is a new line.
        :param x_cord: x-coordinate the image will be inserted at in the current PDF Page
        :param y_cord: y-coordinate the image will be inserted at in the current PDF Page
        :param textcolor: Color of the text.
        :param font: Font of the text. If none is passed, font of the PDF will be taken.
        :param fontsize: Fontsize of the text. If none is passed, content fontsize of the PDF will be taken.
        :return:
        """

        if fontsize == 0:
            fontsize = self.fontsize_content
        if font == "":
            font = self.font_content
        text_configuration = self.canvas.beginText()
        text_configuration.setFont(font, fontsize)
        text_configuration.setFillColor(textcolor)
        text_configuration.setTextOrigin(x_cord * mm, y_cord * mm)
        text_configuration.textLines(text)
        self.canvas.drawText(text_configuration)

    def insert_textfile(self, text_dir: str, text_file: str, x_cord: int, y_cord: int):
        """
        Inserts text from a textfile (.txt) to the current PDF Page. Text can be formatted in XML.

        :param text_dir: Path to the text file directory.
        :param text_file: Name of the text file.
        :param x_cord: x-coordinate the text will be inserted at in the current PDF Page
        :param y_cord: y-coordinate the text will be inserted at in the current PDF Page
        :return:
        """

        if text_file.endswith(".txt"):
            with open(os.path.join(self.project_dir, text_dir, text_file)) as text:
                text = Paragraph(text.read())
                text.wrap(self.right_margin * mm - x_cord * mm, self.top_margin * mm - y_cord * mm)
                text.drawOn(self.canvas, x_cord * mm, y_cord * mm - text.height)

    def insert_textfield(self, x_cord: int, y_cord: int, width: int, height: int, max_length: int = 10000,
                         text: str or list = "", tooltip: str = "",
                         font: str = "", fontsize: int = 0,
                         textcolor: colors = colors.black,
                         fillcolor: colors = colors.white,
                         borderwidth: int = 0,
                         bordercolor: colors = colors.white):
        """
        Inserts a Textfield to the current PDF Page. A textfield can be edited within the PDF Viewer.

        :param x_cord: x-coordinate the textfield will be inserted at in the current PDF Page
        :param y_cord: y-coordinate the textfield will be inserted at in the current PDF Page
        :param width: Width of the textfield in mm
        :param height: Height of the textfield in mm
        :param max_length: Max length of the text that can be written in the PDF Viewer
        :param text: Pre-Text that can be edited in the PDF Viewer
        :param tooltip: Tooltip to be shwon in the PDF viewer
        :param font: Font of the text. If none is passed, font of the PDF will be taken.
        :param fontsize: Fontsize of the text. If none is passed, content fontsize of the PDF will be taken.
        :param textcolor: Color of the text. If none is passed, it's "black"
        :param fillcolor: Background color of the textfield. If none is passed, it's "white"
        :param borderwidth: Width of the textfield border. If none is passed, it's 0.
        :param bordercolor: Color of the textfield border. If none is passed, it's "white".
        :return:
        """

        if fontsize == 0:
            fontsize = self.fontsize_content
        if font == "":
            font = self.font_content
        self.canvas.acroForm.textfield(text,
                                       fieldFlags='multiline',
                                       x=x_cord * mm, y=y_cord * mm,
                                       width=width * mm, height=height * mm,
                                       fontSize=fontsize, fontName=font,
                                       maxlen=max_length,
                                       fillColor=fillcolor,
                                       borderColor=bordercolor,
                                       textColor=textcolor,
                                       borderWidth=borderwidth,
                                       forceBorder=True,
                                       tooltip=tooltip)

    def insert_section_header(self, text: str, y_cord: int, textcolor: str = "black"):
        """
        Insert a section header to the current PDF Page. Consists of a centered text,
        with lines to the left and right.

        :param text: Text / Name of the section header.
        :param y_cord: y-coordinate the section header will be inserted at in the current PDF Page
        :param textcolor: Color of the text. If none is passed, it's "black"
        :return:
        """

        self.canvas.setFont(self.font_bold, 14)
        self.canvas.setFillColor(textcolor)
        self.canvas.drawCentredString(105 * mm, y_cord * mm, text)
        self.canvas.setLineWidth(0.05 * mm)
        self.canvas.line(self.left_margin * mm, (y_cord + 1) * mm, (self.mid_page - len(text) * 1.4) * mm,
                         (y_cord + 1) * mm)
        self.canvas.line((self.mid_page + len(text) * 1.4) * mm, (y_cord + 1) * mm, self.right_margin * mm,
                         (y_cord + 1) * mm)

    def insert_headline(self, text: str, y_cord: int, textcolor: str = "black"):
        """
        Inserts a bolded headline to the current PDF Page.

        :param text: Text / Name of the current Headline
        :param y_cord: y-coordinate the Headline will be inserted at in the current PDF Page
        :param textcolor: Color of the Headline. If none is passed, it's "black"
        :return:
        """

        self.canvas.setFont(self.font_bold, self.fontsize_content + 2)
        self.canvas.setFillColor(textcolor)
        self.canvas.drawString(self.left_margin * mm, y_cord * mm, text)

    def insert_table(self, table_data: pd.DataFrame, x_cord: int, y_cord: int, row_colors: list = None,
                     textcolor: str = "black", linecolor: str = "grey", removeindex: bool = True):
        """
        Inserts a table to the current PDF Page.

        :param table_data: Table as Pandas Dataframe.
        :param x_cord: x-coordinate the Table will be inserted at in the current PDF Page
        :param y_cord: y-coordinate the Table will be inserted at in the current PDF Page
        :param row_colors: Color of the Table's rows as a list. Every list item is one row.
        :param textcolor: Color of the text in the table, not specified by "row_colors". If none is passed, it's "black"
        :param linecolor: Color of the Table's lines
        :param removeindex: Removes the Pandas Dataframe Index if True. If False,
        index will be written bolded at the top of the table
        :return:
        """

        if removeindex:
            table_data.index.rename("", inplace=True)

        datalist = [table_data.reset_index().columns.tolist()] + list(table_data.reset_index().values)
        table = Table(datalist)
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.font_content),
            ('FONTNAME', (0, 0), (0, -1), self.font_bold),
            ('FONTSIZE', (0, 0), (-1, -1), self.fontsize_table),
            ('FONTSIZE', (0, 0), (0, 0), self.fontsize_table + 2),
            ('TEXTCOLOR', (0, 0), (-1, -1), textcolor),
            ('LINEBELOW', (0, 0), (-1, 0), 0.25, linecolor),
            ('ALIGNMENT', (1, 0), (-1, -1), 'CENTER')
        ]))

        if row_colors is not None:
            row = 1
            for color in row_colors:
                table.setStyle(TableStyle([('TEXTCOLOR', (0, row), (-1, -1), color)]))
                row += 1
        table.wrapOn(self.canvas, 0, 0)
        table.drawOn(self.canvas, x_cord * mm, y_cord * mm)

    def create_cover_sheet(self):
        """
        Creates a pre-defined Cover Sheet with Title, Project, Confidentiality,
        Author and Date as defined in the config file

        :return:
        """

        self.canvas.drawImage(os.path.join("layout", self.pictures_dict["logo_cover"]), 10 * mm, 260 * mm,
                              width=(293 / 8) * mm, height=(185 / 8) * mm)

        self.canvas.setFont(self.font_bold, 24)
        self.canvas.drawCentredString(105 * mm, 200 * mm, self.general_dict["title"])

        self.canvas.setFont(self.font_bold, 22)
        self.canvas.drawCentredString(105 * mm, 170 * mm, self.general_dict["project"])

        self.canvas.setFont(self.font_bold, 18)
        self.canvas.drawCentredString(105 * mm, 135 * mm, " - " + self.general_dict["confidentiality"] + " -")

        self.canvas.setFont(self.font_bold, 16)
        self.canvas.drawCentredString(105 * mm, 100 * mm, self.general_dict["author"])

        self.canvas.setFont(self.font_content, 16)
        self.canvas.drawCentredString(105 * mm, 70 * mm, str(self.general_dict["date"]))

        self.canvas.showPage()

    def savepdf(self, openpdf: bool = False):
        """
        Saves the current PDF.

        :param openpdf: Opens the PDF in Standard PDF Viewer, if True.
        :return:
        """

        print("[PDFCreator] Saving PDF to " + str(self.pdf_path))
        self.canvas.save()

        if openpdf:
            os.startfile(self.pdf_path)

    def convert_pdf_to_docx(self, opendocx: bool = False):
        """
        Converts the saved PDF to docx with pdf2docx.

        :param opendocx: Opens the docx in Word, if True.
        :return:
        """

        from pdf2docx import parse

        parse(self.pdf_path,
              os.path.join(self.project_dir, str(self.general_dict["date"]) + " " + self.general_dict["title"] + " " +
                           self.general_dict["project"] + ".docx"))

        if opendocx:
            os.startfile(os.path.join(self.project_dir,
                                      str(self.general_dict["date"]) + " " + self.general_dict["title"] + " " +
                                      self.general_dict[
                                          "project"] + ".docx"))
