import os
import pandas as pd
import openpyxl


fpath = '/home/nachiket/Downloads/SCND_assignment_Excel_18-10-2022.xlsx'

'''
wb = openpyxl.load_workbook(filename=fpath, data_only=False)

s3 = wb.sheetnames[-1]

#df = pd.DataFrame(wb)
df = pd.ExcelFile(fpath)

print(df.sheet_names)

mydf = df.parse('Sheet3')

print(s3)

'''

from koala.ExcelCompiler import ExcelCompiler
from koala.Spreadsheet import Spreadsheet

import databricks.koalas as ks

sp = Spreadsheet(fpath, ignore_hidden=True)
c = ExcelCompiler(fpath)
#sp.dump('xldump.zip')

print(sp.cell_evaluate('Sheet3!AA37'))


