import modulefinder

finder = modulefinder.ModuleFinder()
finder.run_script('pdf.py')

print("Imported modules:")
for name, mod in finder.modules.items():
    print(name)
