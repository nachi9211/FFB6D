'''
from openpyxl import load_workbook
import formulas

#The variable spreadsheet provides the full path with filename to the excel spreadsheet with unevaluated formulae
fpath = '/home/nachiket/Downloads/SCND_assignment_Excel_18-10-2022.xlsx'
outpath = '/home/nachiket/Downloads/'#myoutput.xlsx
xl_model = formulas.ExcelModel().loads(fpath).finish()
xl_model.calculate()
xl_model.write(outpath)

wb = load_workbook(filename=fpath,data_only=True)
ws = wb.active
'''

from xlcalculator import ModelCompiler
from xlcalculator import Model
from xlcalculator import Evaluator

filename = r'/home/nachiket/Downloads/SCND_assignment_Excel_18-10-2022.xlsx'
compiler = ModelCompiler()
new_model = compiler.read_and_parse_archive(filename)
evaluator = Evaluator(new_model)


val1 = evaluator.evaluate('Sheet3!AA44')
print("value 'evaluated' for Sheet3A44: ", val1)
