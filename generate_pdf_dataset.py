import argparse
import random
import sys
from PyPDF4 import PdfFileWriter, PdfFileReader
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import rand_string.rand_string as rand
import os
import io
from random import randrange
import minecart
from fake_data_generator import fake_tax_data

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import tqdm

curr_dir = os.getcwd()
join_dir = lambda *args: os.path.join(*args)

desc = "GENERATING PDFS WITH SYNTHETIC TAX DATA"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-c', '--form_catalogue_dir', type=str, default='Forms_catalogue',
                    help='Directory of the form catalogue which have multiple pdfs in a single form')

parser.add_argument('-l', '--minimum_lines', type=int, default=10,
                    help='A threshold to select the form based on number of lines present in forms')


parser.add_argument('-f', '--filled_pdf_dir', type=str, default="filled_pdf_fake_tax",
                    help='Directory to save the filled pdfs generated')

parser.add_argument('-o', '--original_pdf_dir', type=str, default="original_pdf_fake_tax",
                    help='Directory to save the original pdfs generated')

parser.add_argument('-v', '--variation_pdf', type=int, default=10,
                    help='Total number of variation pdfs to generate from a single pdf')

parser.add_argument('-i', '--input_file', type=str, default="out.txt",
                    help='File containing the valid font color pages')


args = parser.parse_args()


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pdf(form_filename, forms_dir, pdf_name, font_name, font_size, str_length, page_num, num_dyn_str):
    """Inserts a random string of specified length in a given pdf file and saves the new pdf in the current directory

    :param form_filename:         name of the original pdf file
    :param forms_dir:             directory of forms
    :param str_length:            length of string to be inserted. (used when inserting random string)
    :param font_name:             fill the form with this font name
    :param font_size:             fill the form with this font size
    :param page_num:              pdf page to insert dynamic strings
    :param num_dyn_str:          total number of dynamic strings to be inserted. (used when inserting random string)
    """
    pdf_filled = pdf_name + 'filled'
    pdf_original = pdf_name + 'original'

    # register the downloaded font
    pdfmetrics.registerFont(TTFont(font_name, os.getcwd() + "/Fonts/{}.ttf".format(font_name)))
    # pdfmetrics.registerFont(TTFont('TimesNewRomanPSMT', os.getcwd() + "/Fonts/{}.ttf".format("TimesNewRomanPSMT")))
    # move to the beginning of the BytesIO buffer
    packet = io.BytesIO()

    # create a new PDF with reportlab
    can = canvas.Canvas(packet, pagesize=letter)

    can.translate(inch, inch)
    can.setFont(font_name, font_size)
    can.setFillColorRGB(0, 0, 0)  ## black text font color by default


    list_random_strings = []

    # fake text data return 39 random strings, filling pdf with 39x3 random strings
    print("Inserting synthetic tax data from faker library into pdf")
    co_x = 130
    co_y = 250
    k = 0
    for _ in tqdm.tqdm(range(3)): # fake library generates 39 different type of information (loop 3 times )
        fake_data = fake_tax_data()
        list_random_strings.extend(fake_data)
        for text in fake_data:
            can.drawString(randrange(co_x*k, co_x*(k+1)), randrange(750), text)
        k += 1

# UNCOMMENT TO INSERT RANDOM STRINGS INTO PDF
    # print("Generating random strings")
    # for _ in tqdm.tqdm(range(1, (num_dyn_str)//2 + 1)):
    #
    #     text1 = rand.RandString("upperlower", randrange(str_length)+1).strip()
    #     text2 = rand.RandString("alphanumerical", randrange(str_length) + 1).strip()
    #
    #     list_random_strings.append(text1)
    #     list_random_strings.append(text2)
    #     can.drawString(randrange(450), randrange(750), text1)
    #     can.drawString(randrange(450), randrange(750), text2)


     # List of available fonts by default and registered
    if font_name in can.getAvailableFonts():
        print("Found the font {}".format(font_name))
    else:
        print(" -------------------------Font Not Found------------------------- ")


    can.showPage()
    can.save()

    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    # creating an object
    file_reader = open(forms_dir + '/' + form_filename, 'rb')

    # read your existing PDF
    existing_pdf = PdfFileReader(file_reader)

    output = PdfFileWriter()
    page = existing_pdf.getPage(page_num)

    page.mergePage(new_pdf.getPage(0))
    output.addPage(page)

    # directory to save original and generated filled pdf's
    check_dir(args.filled_pdf_dir)
    check_dir(args.original_pdf_dir)

    output_stream2 = open(curr_dir + '/filled_pdf_fake_tax/' + pdf_filled + '.pdf', "wb")


    output.write(output_stream2)
    output_stream2.close()
    print("\n----Pdf successfully generated with fake tax data-----")

    extract_current_page(page_num, form_filename, pdf_original)
    print("\n----saved the original Pdf as well-----")


def extract_current_page(page_num, filename, original_name):
    """

    :param page_num: extract the particular page with page_num
    :param filename: filename to extract the particular page
    :return: return  filename with original appended to the name, also saves the page in cwd().
    """
    with open(forms_dir + '/' + filename, 'rb') as infile:
        reader = PdfFileReader(infile)
        writer = PdfFileWriter()
        writer.addPage(reader.getPage(page_num))

        pdfs_name = os.getcwd() + '/original_pdf_fake_tax/' + original_name + '.pdf'
        with open(pdfs_name, 'wb') as outfile:
            writer.write(outfile)


def valid_line_pages(filename, dir_path, minimum_lines):
    """
    Check whether a form is valid or not based on number of lines present in it
    :param filename:
    :param dir_path:
    :param minimum_lines:
    :return: list of valid pdf pages on a form
    """
    count = 0
    valid_line_page_list = []
    A4_SIZE = [0.0, 0.0, 595.28, 841.89]

    pdffile = open(dir_path + '/' + filename, 'rb')
    doc = minecart.Document(pdffile)
    page_number = 0
    print(f'\nParsing {filename} for finding the total number of lines')
    pdf_file_reader = PdfFileReader(open(dir_path + '/' + filename, 'rb'))
    for p in doc.iter_pages():
        page_size = list(pdf_file_reader.getPage(page_number).mediaBox)
        for shape in p.shapes:
            if not shape.path:
                continue
            else:
                count += 1
        print("Total lines on page number {} are {}".format(page_number, count))

        # converting the object in list to float data type
        page_size = [float(p) for p in page_size]

        if page_size == A4_SIZE and count >= minimum_lines:
            valid_line_page_list.append(page_number)
        count = 0
        page_number += 1

    return valid_line_page_list



import ast
# File containing the valid font color pages, out.txt
forms = []
font_name = []
font_size = []
valid_font_color_pages = []
in_file = args.input_file
file_object = open(join_dir(curr_dir, in_file))
lines = file_object.readlines()
for line in lines:
    data = line.split('#')
    forms.append(data[0].strip())
    font_name.append(data[1].strip())
    font_size.append(data[2].strip())

    # data[3] contains list of valid color pages(exclude pdfs containing colored text or shape)
    # in the form of string (ast python package rebuilds list from string)
    valid_font_color_pages.append(ast.literal_eval(data[3].strip()))

ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n//10 % 10 !=1) *(n%10<4)*n%10::4])

forms_dir = join_dir(curr_dir, args.form_catalogue_dir)

a = 0
minimum_lines = args.minimum_lines
total_variation_pdf = args.variation_pdf
pdf_counter = 1
for form, font, size, valid_font_color_page in tqdm.tqdm(zip(forms, font_name, font_size, valid_font_color_pages)):
    valid_pages = list(set(valid_font_color_page) & set(valid_line_pages(form, forms_dir, minimum_lines)))
    a += len(valid_pages)
    print("Momentarily total valid pages pdfs are {}".format(a))
# print("Total valid pages in the data set with black color and number of lines filter are {}".format(a))

    for page in valid_pages:
        pdf_counter_formatted = str(pdf_counter).zfill(4)
        for i in tqdm.tqdm(range(1, total_variation_pdf)):
            pdf_var_counter = str(i).zfill(2)
            pdf_name = "{}_{}_".format(pdf_counter_formatted, pdf_var_counter)
            save_pdf(form, forms_dir, pdf_name, font, float(size), 20, page, 110)
            break
        pdf_counter += 1
    print(f'Generated {total_variation_pdf} variations pdf pages of all relevant pdf pages from form {form}\n\n')
    break



