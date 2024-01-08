# -*- coding: utf-8 -*-
"""
    scisys.io.pdf.writer
    ~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import os
import pandas as pd
import datetime as dt

from hashlib import sha1
from reportlab.lib import colors
from reportlab.lib.utils import Image
from reportlab.lib.units import mm
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import NextPageTemplate, PageBreak, Spacer, Table, TableStyle, Paragraph
from reportlab.pdfbase.pdfmetrics import stringWidth
from corsys import Configurable, Configurations
from .template import PdfDocTemplate
from .styles import PdfStyles

DATE_FORMAT: str = "%d.%m.%Y"


class PdfWriter(Configurable):

    SECTION = 'PDF'

    # noinspection PyTypeChecker
    @classmethod
    def read(cls, system, configs: Configurations, conf_file: str = 'pdf.cfg', section: str = SECTION,
             confidential: bool = False) -> PdfWriter:
        config_args = {
            'require': False
        }
        if configs.has_section(section):
            config_args.update(configs.items(section))

        sections = {s: configs.items(s) for s in configs.sections() if s.startswith(cls.SECTION)}

        # TODO: Verify usefulness of this section just for logo locations
        if configs.has_section('Images'):
            sections['Images'] = {k: v for k, v in configs['Images'].items() if k.startswith('logo_')}

        configs = Configurations.from_configs(configs, conf_file, **config_args)
        for section, config_params in sections.items():
            config_section = section[len(cls.SECTION)+1:]
            if not configs.has_section(config_section):
                configs.add_section(config_section)
            for key, val in config_params.items():
                configs.set(config_section, key, val)

        creation_date = configs.get(Configurations.GENERAL, 'date', fallback=dt.datetime.now().strftime(DATE_FORMAT))
        filename = configs.get(Configurations.GENERAL, 'file', fallback=system.id)
        filepath = os.path.join(system.configs.dirs.data, f"{filename}.pdf")

        return cls(configs, filepath, creation_date, confidential)

    def __init__(self, configs: Configurations, filename: str, creation_date: str, confidential: bool = False):
        super().__init__(configs)
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except PermissionError:
            filename = filename.replace('.pdf', '(2).pdf')
        self._filename = filename
        self._creation_date = creation_date

        # TODO: Make a list of logos configurable
        logo_cover = configs.get('Images', 'logo_cover', fallback=None)
        logo_header = configs.get('Images', 'logo_header', fallback=None)

        self._styles = PdfStyles(configs)
        self._document = PdfDocTemplate(self._styles, filename, creation_date,
                                        self._img_dir, logo_cover, logo_header,
                                        confidential)
        self._header_level = [0] * len([s for s in self._styles.byName.keys() if s.startswith('Heading')])
        self._content = []

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self._img_dir = os.path.join(configs.dirs.lib, 'images')

    def add_cover(self,
                  title: str,
                  subtitle: str,
                  author: str):
        """
        Creates a pre-defined Cover Sheet with Title, Project, Confidentiality,
        Author and Date as defined in the config file

        :return:
        """
        self._content.append(NextPageTemplate('Cover'))

        # Center the title
        title_paragraph = Paragraph(title, self._styles['Title'])
        _, title_height = title_paragraph.wrap(self._document.width, self._document.height)
        title_position = self._document.height/2 - title_height

        self._content.append(Spacer(1, title_position))

        self._content.append(title_paragraph)
        self._content.append(Paragraph(subtitle, self._styles['Subtitle']))

        author_paragraph = Paragraph(author, self._styles['Cover'])
        _, author_height = author_paragraph.wrap(self._document.width, self._document.height)

        # Author and date centered in lower quarter
        self._content.append(Spacer(1, title_position/2 - author_height))

        self._content.append(author_paragraph)

        self._content.append(Spacer(1, 12))
        self._content.append(Paragraph(self._creation_date, self._styles['Cover']))

        self._content.append(NextPageTemplate('Normal'))
        self._content.append(PageBreak())

    def add_table_of_content(self):
        table_of_content = TableOfContents(dotsMinLevel=0)
        table_of_content.levelStyles = [
            self._styles['TableOfContents']
        ]
        self._content.append(Paragraph('Table of Contents', self._styles['Heading3']))
        self._content.append(table_of_content)
        self._content.append(PageBreak())

    def add_page_break(self):
        self._content.append(PageBreak())

    def add_paragraph(self, text: str, style: str = 'Normal'):
        self._content.append(Paragraph(text, self._styles[style]))

    def add_header(self, header: str, level: int = 1):
        style = self._styles[f'Heading{level}']

        # Create an autoincrement heading number
        self._header_level[level - 1] += 1
        self._header_level[level:] = [0] * (len(self._header_level) - level)
        header_number = '.'.join(str(i) for i in self._header_level[:level])
        if level == 1:
            header_number += '.'

        # TODO: Find better solution to mimic a tab for header numbers
        while stringWidth(header_number, style.fontName, style.fontSize) < 10*mm:
            header_number += '\xa0'

        # Create a unique bookmark name
        header_bookmark = sha1(f'{header_number} {header} {style.name}'.encode('utf-8')).hexdigest()

        # Modify the paragraph text to include an anchor point with the bookmark
        header = Paragraph(f'<a name="{header_bookmark}"/>{header_number} {header}', style)

        # Store the bookmark name on the flowable so afterFlowable can see this
        header._bookmarkName = header_bookmark

        self._content.append(header)

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
            fontsize = self.font_size_content
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
            fontsize = self.font_size_content
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
        self.canvas.setFont(self.font_bold, self.font_size_content + 2)
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
            ('FONTSIZE', (0, 0), (-1, -1), self.font_size_table),
            ('FONTSIZE', (0, 0), (0, 0), self.font_size_table + 2),
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

    # noinspection PyShadowingBuiltins
    def save(self, open: bool = False):
        """
        Saves the current PDF.

        :param open: Opens the PDF in Standard PDF Viewer, if True.
        :return:
        """
        self._document.prepareBuild(self._content)
        self._document.multiBuild(self._content)
        if open:
            os.startfile(self._filename)

    # noinspection PyUnresolvedReferences, PyPackageRequirements, PyShadowingBuiltins
    def save_as_docx(self, open: bool = False):
        """
        Converts the saved PDF to docx with pdf2docx.

        :param open: Opens the docx in Word, if True.
        :return:
        """
        filename = self._filename.replace('.pdf', '.docx')

        import pdf2docx as docx
        docx.parse(filename)

        if open:
            os.startfile(filename)
