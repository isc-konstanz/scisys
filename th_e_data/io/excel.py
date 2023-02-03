# -*- coding: utf-8 -*-
"""
    th-e-sim.io.excel
    ~~~~~~~~~~~~~~~~~


"""
import os
import logging
import warnings
import pandas as pd

from openpyxl.utils import get_column_letter
from openpyxl.styles import Border, Font, Side
from tables import NaturalNameWarning
from copy import copy

warnings.filterwarnings('ignore', category=NaturalNameWarning)
logger = logging.getLogger(__name__)


def write_excel(summary, data_frames, data_dir: str = 'data'):
    border_side = Side(border_style=None)
    border = Border(top=border_side,
                    right=border_side,
                    bottom=border_side,
                    left=border_side)

    summary_file = os.path.join(data_dir, 'summary.xlsx')
    with pd.ExcelWriter(summary_file, engine='openpyxl') as summary_writer:
        summary.to_excel(summary_writer, sheet_name='Summary', float_format="%.2f", encoding='utf-8-sig')
        summary_book = summary_writer.book

        for data_key, data in data_frames.items():
            data.to_excel(summary_writer, sheet_name=data_key, encoding='utf-8-sig')
            data_sheet = summary_book[data_key]
            for data_column in range(1, len(data_sheet[1])):
                data_column_value = data_sheet[1][data_column].value
                if data_column_value is not None:
                    data_column_width = len(data_column_value) + 2
                    data_sheet.column_dimensions[get_column_letter(data_column + 1)].width = \
                        data_column_width
                data_sheet[1][data_column].border = border

        # Set column width and header coloring
        for summary_sheet in summary_book:
            if summary_sheet.title == 'Summary':
                summary_sheet.delete_rows(3, 1)

            summary_index_width = 0
            for summary_row in summary_sheet:
                summary_row[0].border = border
                summary_index_width = max(summary_index_width, len(str(summary_row[0].value)))

            summary_sheet.column_dimensions[get_column_letter(1)].width = summary_index_width + 2

            summary_header_len = len(summary.columns.levels)
            summary_header_font = Font(name="Calibri Light", size=12, color='333333')
            for summary_column in range(len(summary_sheet[summary_header_len])):
                for summary_header_row in range(1, summary_header_len):
                    if '\n' in str(summary_sheet[summary_header_row][summary_column].value):
                        summary_header_alignment = copy(summary_sheet[summary_header_row][summary_column].alignment)
                        summary_header_alignment.wrapText = True
                        # summary_header_alignment.vertical = 'center'
                        # summary_header_alignment.horizontal = 'center'
                        summary_sheet[summary_header_row][summary_column].alignment = summary_header_alignment
                        summary_sheet.row_dimensions[summary_header_row].height = 33

                    summary_sheet[summary_header_row][summary_column].font = summary_header_font
                    summary_sheet[summary_header_row][summary_column].border = border

                summary_header_width = 0
                for summary_header in range(1, summary_header_len+1):
                    summary_header_width = max(summary_header_width,
                                               len(str(summary_sheet[summary_header+1][summary_column].value)))
                    summary_sheet[summary_header][summary_column].border = border
                summary_sheet.column_dimensions[get_column_letter(summary_column+1)].width = summary_header_width + 2
