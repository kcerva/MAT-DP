
import tkinter as tk
from tkinter import font as tkfont
import math
import scipy.stats as sci
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import copy
from SALib.sample import saltelli
from SALib.analyze import sobol

rcParams.update({'figure.autolayout': True})
sobolCount = 1000
#Initialize Global Variables and Dictionaries
emissions = []
total_mat_emissions_per_year = {}
total_vehicle_emissions_per_year = {}
total_driving_emissions_per_year = {}
tot_mat_amounts_per_year = {}
tot_mat_emissions = {}
total_vehicle_emissions_per_year = {}
tot_mat_amounts = {}
driving_emissions = {}
pageName = ""
country_list = {}
transportGrowth = 0.06
totalYears = 20
pal = sns.color_palette("hls", 12)

class FrameApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("1000x800")
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.reg_font = tkfont.Font(family='Helvetica', size=10)

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        self.container = tk.Frame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (ScenOne, ScenTwo, ScenThree, ResultPage):
            self.init_frame(F)

        self.show_frame("ScenOne")

    def init_frame(self, page_name):
        p = page_name.__name__
        frame = page_name(parent=self.container, controller=self)
        self.frames[p] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class ScenOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the Business as Usual Scenario", font=controller.title_font)
        label.grid(row=0, column=1, sticky='N', pady=10)
        label2 = tk.Label(self, text="enter passenger km here:", font=controller.reg_font)
        label2.grid(row=1, column=0, sticky='W', pady=1)
        self.entry_1 = tk.Entry(self)
        self.entry_1.grid(row=1, column=1, pady=1)
        self.entry_1.insert(0, 56200000000)

        self.labels = {}
        self.pentries = {}
        self.fentries = {}
        self.ientries = {}
        self.elecentries = {}
        self.hybridentries = {}
        elecVals= [0, 0, 0, 0, 0, 0, 0, 0]
        infraVals= [300, 2]
        rel_count = 0
        for counter, F in enumerate(TransportModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of pkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.pentries[str(name)] = tk.Entry(self)
            self.pentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.pentries[str(name)].insert(0, F.defaultPer)
            self.labels[str(name)+"Elec"] = tk.Label(self, text="enter % electric of " + name + " here:", font=controller.reg_font)
            self.labels[str(name)+"Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter])

            rel_count += 2
        label_3 = tk.Label(self, text="enter tonne km here:", font=controller.reg_font)
        label_3.grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
        self.entry_2 = tk.Entry(self)
        self.entry_2.grid(row=rel_count + 2, column=1, pady=1)
        self.entry_2.insert(0, 14149000000)
        rel_count += 1
        for counter, F in enumerate(FreightModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of tkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.fentries[str(name)] = tk.Entry(self)
            self.fentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.fentries[str(name)].insert(0, F.defaultPer)
            self.labels[str(name) + "Elec"] = tk.Label(self, text="enter % electric of " + name + " here:",
                                                       font=controller.reg_font)
            self.labels[str(name) + "Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter+5])
            rel_count += 2
        for counter, F in enumerate(InfraModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " amount here:" + " (" + F.unit + ")",
                                              font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
            self.ientries[str(name)] = tk.Entry(self)
            self.ientries[str(name)].grid(row=rel_count + 2, column=1, pady=1)
            self.ientries[str(name)].insert(0, infraVals[counter])
            rel_count += 1
        country_sel = tk.Label(self, text="Select country here:", font=controller.reg_font)
        country_sel.grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
        self.variable = tk.StringVar(self)
        self.variable.set('Uganda')
        self.country_list = tk.OptionMenu(self, self.variable, 'Uganda')
        self.country_list.grid(row=rel_count + 2, column=1, pady=1)
        rel_count += 1
        gridFLabel = tk.Label(self, text="Select Grid Emissions Factor:", font=controller.reg_font)
        gridFLabel.grid(row=rel_count + 2, column=0, pady=1)
        self.gridFactor = tk.Entry(self)
        self.gridFactor.grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
        self.gridFactor.insert(0, .772)
        rel_count += 1
        get_results = tk.Button(self, text="Get Emissions",
                                command=lambda: get_emissions(self, 0))

        get_results.grid(row=rel_count + 2, column=0, pady=1)
        scen_two = tk.Button(self, text="Go to Electric Scenario",
                                command=lambda: self.controller.init_frame(ScenTwo))

        scen_two.grid(row=rel_count + 2, column=1, pady=1)
        scen_three = tk.Button(self, text="Go to Public and Rail Scenario",
                                command=lambda: self.controller.init_frame(ScenThree))

        scen_three.grid(row=rel_count + 2, column=2, pady=1)


    def get_entries(self):
        userPkm = float(self.entry_1.get())
        userTkm = float(self.entry_2.get())
        percentsPkm = {}
        percentsTkm = {}
        infraAmounts = {}
        perElectric = {}
        totalP = 0
        totalT = 0
        gridFactor = float(self.gridFactor.get())
        for key in self.pentries:
            percentsPkm[key] = float(self.pentries[key].get())
            totalP += float(self.pentries[key].get())
        for key in self.fentries:
            percentsTkm[key] = float(self.fentries[key].get())
            totalT += float(self.fentries[key].get())
        for key in self.ientries:
            infraAmounts[key] = float(self.ientries[key].get())
        userCountry = country_list[self.variable.get()]
        for key in self.elecentries:
            perElectric[key] = float(self.elecentries[key].get())
        if totalP != 100 or totalT != 100:
            raise Exception("Percentages do not add up to 100")
        return userPkm, userTkm, userCountry, percentsPkm, percentsTkm, infraAmounts, perElectric, gridFactor


class ScenTwo(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the Electric Emphasis Scenario", font=controller.title_font)
        label.grid(row=0, column=1, sticky='N', pady=10)
        label2 = tk.Label(self, text="enter passenger km here:", font=controller.reg_font)
        label2.grid(row=1, column=0, sticky='W', pady=1)
        self.entry_1 = tk.Entry(self)
        self.entry_1.grid(row=1, column=1, pady=1)
        self.entry_1.insert(0, 56200000000)

        self.labels = {}
        self.pentries = {}
        self.fentries = {}
        self.ientries = {}
        self.elecentries = {}
        self.hybridentries = {}
        elecVals= [60, 0, 30, 60, 60, 40, 30, 0]
        perVals= [13.9, .01, 17.5, 59.4, 9.19, 19.1, 76.3, 4.6]
        infraVals= [300, 2]
        rel_count = 0
        for counter, F in enumerate(TransportModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of pkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.pentries[str(name)] = tk.Entry(self)
            self.pentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.pentries[str(name)].insert(0, perVals[counter])
            self.labels[str(name)+"Elec"] = tk.Label(self, text="enter % electric of " + name + " here:", font=controller.reg_font)
            self.labels[str(name)+"Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter])

            rel_count += 2
        label_3 = tk.Label(self, text="enter tonne km here:", font=controller.reg_font)
        label_3.grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
        self.entry_2 = tk.Entry(self)
        self.entry_2.grid(row=rel_count + 2, column=1, pady=1)
        self.entry_2.insert(0, 14149000000)
        rel_count += 1
        for counter, F in enumerate(FreightModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of tkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.fentries[str(name)] = tk.Entry(self)
            self.fentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.fentries[str(name)].insert(0, perVals[counter+5])
            self.labels[str(name) + "Elec"] = tk.Label(self, text="enter % electric of " + name + " here:",
                                                       font=controller.reg_font)
            self.labels[str(name) + "Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter+5])
            rel_count += 2
        for counter, F in enumerate(InfraModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " amount here:" + " (" + F.unit + ")",
                                              font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
            self.ientries[str(name)] = tk.Entry(self)
            self.ientries[str(name)].grid(row=rel_count + 2, column=1, pady=1)
            self.ientries[str(name)].insert(0, infraVals[counter])
            rel_count += 1
        self.variable = tk.StringVar(self)
        self.variable.set('Uganda')
        self.country_list = tk.OptionMenu(self, self.variable, 'Uganda')
        self.country_list.grid(row=rel_count + 2, column=1, pady=1)

        rel_count += 1
        gridFLabel = tk.Label(self, text="Select Grid Emissions Factor:", font=controller.reg_font)
        gridFLabel.grid(row=rel_count + 2, column=0, pady=1)
        self.gridFactor = tk.Entry(self)
        self.gridFactor.grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
        self.gridFactor.insert(0,.55)
        rel_count += 1
        get_results = tk.Button(self, text="Get Emissions",
                                command=lambda: get_emissions(self, 5))

        get_results.grid(row=rel_count + 2, column=0, pady=1)
        scen_two = tk.Button(self, text="Go to BAU Scenario",
                                command=lambda: self.controller.init_frame(ScenOne))

        scen_two.grid(row=rel_count + 2, column=1, pady=1)
        scen_three = tk.Button(self, text="Go to Public and Rail Scenario",
                                command=lambda: self.controller.init_frame(ScenThree))

        scen_three.grid(row=rel_count + 2, column=2, pady=1)

    def get_entries(self):
        userPkm = float(self.entry_1.get())
        userTkm = float(self.entry_2.get())
        percentsPkm = {}
        percentsTkm = {}
        infraAmounts = {}
        perElectric = {}
        totalP = 0
        totalT = 0
        gridFactor = float(self.gridFactor.get())
        for key in self.pentries:
            percentsPkm[key] = float(self.pentries[key].get())
            totalP += float(self.pentries[key].get())
        for key in self.fentries:
            percentsTkm[key] = float(self.fentries[key].get())
            totalT += float(self.fentries[key].get())
        for key in self.ientries:
            infraAmounts[key] = float(self.ientries[key].get())
        userCountry = country_list[self.variable.get()]
        for key in self.elecentries:
            perElectric[key] = float(self.elecentries[key].get())
        if totalP != 100 or totalT != 100:
            raise Exception("Percentages do not add up to 100")
        return userPkm, userTkm, userCountry, percentsPkm, percentsTkm, infraAmounts, perElectric, gridFactor


class ScenThree(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the Public Transport and Rail Scenario", font=controller.title_font)
        label.grid(row=0, column=1, sticky='N', pady=10)
        label2 = tk.Label(self, text="enter passenger km here:", font=controller.reg_font)
        label2.grid(row=1, column=0, sticky='W', pady=1)
        self.entry_1 = tk.Entry(self)
        self.entry_1.grid(row=1, column=1, pady=1)
        self.entry_1.insert(0, 56200000000)

        self.labels = {}
        self.pentries = {}
        self.fentries = {}
        self.ientries = {}
        self.elecentries = {}
        self.hybridentries = {}
        # elecVals= [20, 0, 10, 20, 20, 10, 5, 0]
        elecVals= [0, 0, 0, 0, 0, 0, 0, 0]
        perVals= [2.8, 10, 20, 59.8, 7.4, 10, 40, 50]
        infraVals= [300, 80]
        rel_count = 0
        for counter, F in enumerate(TransportModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of pkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.pentries[str(name)] = tk.Entry(self)
            self.pentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.pentries[str(name)].insert(0, perVals[counter])
            self.labels[str(name)+"Elec"] = tk.Label(self, text="enter % electric of " + name + " here:", font=controller.reg_font)
            self.labels[str(name)+"Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter])

            rel_count += 2
        label_3 = tk.Label(self, text="enter tonne km here:", font=controller.reg_font)
        label_3.grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
        self.entry_2 = tk.Entry(self)
        self.entry_2.grid(row=rel_count + 2, column=1, pady=1)
        self.entry_2.insert(0, 14149000000)
        rel_count += 1
        for counter, F in enumerate(FreightModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " % of tkm here:", font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
            self.fentries[str(name)] = tk.Entry(self)
            self.fentries[str(name)].grid(row=rel_count + 2, column=2, pady=1)
            self.fentries[str(name)].insert(0, perVals[counter+5])
            self.labels[str(name) + "Elec"] = tk.Label(self, text="enter % electric of " + name + " here:",
                                                       font=controller.reg_font)
            self.labels[str(name) + "Elec"].grid(row=rel_count + 1, column=3, sticky=tk.W, pady=1)
            self.elecentries[str(name)] = tk.Entry(self)
            self.elecentries[str(name)].grid(row=rel_count + 2, column=3, pady=1)
            self.elecentries[str(name)].insert(0, elecVals[counter+5])
            rel_count += 2
        for counter, F in enumerate(InfraModes):
            name = str(F.name)
            self.labels[str(name)] = tk.Label(self, text="enter " + name + " amount here:" + " (" + F.unit + ")",
                                              font=controller.reg_font)
            self.labels[str(name)].grid(row=rel_count + 2, column=0, sticky=tk.W, pady=1)
            self.ientries[str(name)] = tk.Entry(self)
            self.ientries[str(name)].grid(row=rel_count + 2, column=1, pady=1)
            self.ientries[str(name)].insert(0, infraVals[counter])
            rel_count += 1
        self.variable = tk.StringVar(self)
        self.variable.set('Uganda')
        self.country_list = tk.OptionMenu(self, self.variable, 'Uganda')
        self.country_list.grid(row=rel_count + 2, column=1, pady=1)

        rel_count += 1
        gridFLabel = tk.Label(self, text="Select Grid Emissions Factor:", font=controller.reg_font)
        gridFLabel.grid(row=rel_count + 2, column=0, pady=1)
        self.gridFactor = tk.Entry(self)
        self.gridFactor.grid(row=rel_count + 2, column=1, sticky=tk.W, pady=1)
        self.gridFactor.insert(0,.772)
        rel_count += 1
        get_results = tk.Button(self, text="Get Emissions",
                                command=lambda: get_emissions(self, 10))

        get_results.grid(row=rel_count + 2, column=0, pady=1)
        scen_two = tk.Button(self, text="Go to BAU Scenario",
                                command=lambda: self.controller.init_frame(ScenOne))

        scen_two.grid(row=rel_count + 2, column=1, pady=1)
        scen_three = tk.Button(self, text="Go to Electric Scenario",
                                command=lambda: self.controller.init_frame(ScenTwo))

        scen_three.grid(row=rel_count + 2, column=2, pady=1)

    def get_entries(self):
        userPkm = float(self.entry_1.get())
        userTkm = float(self.entry_2.get())
        percentsPkm = {}
        percentsTkm = {}
        infraAmounts = {}
        perElectric = {}
        gridFactor = float(self.gridFactor.get())
        totalP = 0
        totalT = 0
        for key in self.pentries:
            percentsPkm[key] = float(self.pentries[key].get())
            totalP += float(self.pentries[key].get())
        for key in self.fentries:
            percentsTkm[key] = float(self.fentries[key].get())
            totalT += float(self.fentries[key].get())
        for key in self.ientries:
            infraAmounts[key] = float(self.ientries[key].get())
        userCountry = country_list[self.variable.get()]
        for key in self.elecentries:
            perElectric[key] = float(self.elecentries[key].get())
        if totalP != 100 or totalT != 100:
            raise Exception("Percentages do not add up to 100")
        return userPkm, userTkm, userCountry, percentsPkm, percentsTkm, infraAmounts, perElectric, gridFactor


def get_emissions(page, figVal):

    global emissions
    global total_mat_emissions_per_year
    total_mat_emissions_per_year = {}
    global tot_mat_amounts_per_year
    tot_mat_amounts_per_year = {}
    global total_driving_emissions_per_year
    total_driving_emissions_per_year = {}
    emissions = []
    global tot_mat_emissions
    tot_mat_emissions = {}
    global tot_mat_amounts
    tot_mat_amounts = {}
    global driving_emissions
    driving_emissions = {}
    global total_vehicle_emissions_per_year
    total_driving_emissions_per_year = {}
    pkm, tkm, country, percentsPkm, percentsTkm, infraAmounts, perElectric, gridFactor = page.get_entries()
    problem = {
        'num_vars': 5,
        'names': ['Pkm Percent Split', 'Tkm Percent Split', 'Grid Intensity', 'Tech Advances', 'Percent Electric'],
        'bounds': [[0, 9999], [0, 9999], [gridFactor * .8, gridFactor * 1.2],
                   [0, 1],
                   [.8, 1.2]]
    }

    param_values = saltelli.sample(problem, sobolCount, calc_second_order=False)
    pPkmMC = np.random.dirichlet([i * 5 for i in list(percentsPkm.values())], 9999).transpose()

    pTkmMC = np.random.dirichlet([i * 5 for i in list(percentsTkm.values())], 9999).transpose()

    perPkmMC = {}
    perTkmMC = {}
    emissions = np.zeros([param_values.shape[0]])
    EmissionsPerYear = np.zeros([param_values.shape[0], totalYears])
    print(EmissionsPerYear.shape)
    drivingModes = TransportModes+FreightModes
    combineModes= TransportModes+FreightModes+InfraModes
    for obj in drivingModes:
        total_driving_emissions_per_year[obj.name] = np.zeros([totalYears, param_values.shape[0]])
        total_vehicle_emissions_per_year[obj.name] = np.zeros([totalYears, param_values.shape[0]])

    for obj in combineModes:
        for key in obj.materialList:
            if key not in total_mat_emissions_per_year:
                total_mat_emissions_per_year[key] = np.zeros([totalYears, param_values.shape[0]])
                tot_mat_amounts_per_year[key] = np.zeros([totalYears, param_values.shape[0]])

    for j in range(param_values.shape[0]):
        pkmNew = copy.copy(pkm)
        tkmNew = copy.copy(tkm)
        count = 0
        perCount = 0
        for key in percentsPkm:
            perPkmMC[key] = pPkmMC[perCount][int(round(param_values[j][0]))] * 100
            perCount += 1
        count += 1
        perCount= 0
        for key in percentsTkm:
            perTkmMC[key] = pTkmMC[perCount][int(round(param_values[j][1]))] * 100
            perCount += 1

        for i in range(totalYears):

            driving_emissions = {}
            tot_mat_amounts = {}
            tot_mat_emissions = {}
            pkmNew = (pkmNew*(1+transportGrowth))
            tkmNew = (tkmNew*(1+transportGrowth))

            for obj in TransportModes:
                total_co2, mat_emissions, driv_emissions = obj.get_emissions(pkmNew, perPkmMC, (.482+((param_values[j][2]-.482)/20)*i),
                                                                             perElectric[obj.name] * param_values[j][4]* i/20, i, j, param_values[j][3])
                emissions[j] += total_co2
                # print(str(i)+" years"+ str(j))
                EmissionsPerYear[j][i] += total_co2
                total_vehicle_emissions_per_year[obj.name][i][j] = total_co2
                for key in mat_emissions:
                    total_mat_emissions_per_year[key][i][j] += mat_emissions[key][0]
                for key in obj.mat_amounts:
                    tot_mat_amounts_per_year[key][i][j] = obj.mat_amounts[key]
                driving_emissions[obj.name] = driv_emissions
                total_driving_emissions_per_year[obj.name][i][j] = driv_emissions

            for obj in FreightModes:
                total_co2, mat_emissions, driv_emissions = obj.get_emissions(tkmNew, perTkmMC, (.482+((param_values[j][2]-.482)/20)*i),
                                                                             perElectric[obj.name] * param_values[j][4]* i/20, i, j, param_values[j][3])
                emissions[j] += total_co2
                EmissionsPerYear[j][i] += total_co2
                total_vehicle_emissions_per_year[obj.name][i][j] = total_co2
                for key in mat_emissions:
                    total_mat_emissions_per_year[key][i][j] += mat_emissions[key][0]
                for key in obj.mat_amounts:
                    tot_mat_amounts_per_year[key][i][j] = obj.mat_amounts[key]
                total_driving_emissions_per_year[obj.name][i][j] = driv_emissions
            count = 1
            for obj in InfraModes:
                total_co2, mat_emissions = obj.get_emissions(infraAmounts[obj.name], (.482+((param_values[j][2]-.482)/20)*i),i,j)
                count += 1
                emissions[j] += total_co2
                EmissionsPerYear[j][i] += total_co2
                for key in mat_emissions:
                    total_mat_emissions_per_year[key][i][j] += mat_emissions[key][0]
                for key in obj.mat_amounts:
                    tot_mat_amounts_per_year[key][i][j] = obj.mat_amounts[key]
    matEmSobol = copy.copy(emissions)
    # plt.figure(8+figVal)
    plt.figure(8)
    palED = sns.color_palette("Blues_r")
    plt.hist(matEmSobol, color=palED[1], edgecolor=palED[0],
             bins='auto')
    # Add labels
    print("mean of EACH year")
    print(EmissionsPerYear.mean(axis=0))

    print("Standard Deviation of Total Emissions:"+str(np.std(emissions)))
    print("Mean of Total Emissions:"+str(np.mean(emissions)))
    plt.title('Histogram of total emissions')
    plt.xlabel("Kg of CO$_{2e}$")
    plt.ylabel("frequency")

    Si = sobol.analyze(problem, matEmSobol, calc_second_order=False, print_to_console=True)
    plt.figure(7+figVal)
    plt.clf()
    width=.4
    plt.bar(np.arange(len(Si['S1']))-.2, Si['S1'], width, yerr=Si['S1_conf'],
            color=palED[1], ecolor=palED[0], capsize=5, alpha=0.9, align='center')
    plt.bar(np.arange(len(Si['ST']))+.2, Si['ST'], width, yerr=Si['ST_conf'],
            color=palED[2], ecolor=palED[0], capsize=5, alpha=0.9, align='center')
    plt.xticks(range(len(Si['S1'])), problem['names'], rotation=60)
    plt.legend(['Individual Sobol Index','Total Sobol Index'])
    plt.title("Sobol Indices of Inputs")

    plt.ylabel('Sobol Indices')

    page.controller.init_frame(ResultPage)


class ResultPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        global pageName
        label = tk.Label(self, text="This is emissions: ")
        label.grid(row=0, column=0, sticky='W', pady=10)
        if len(emissions)>1:
            output_em = tk.Label(self, text=str(format(np.mean(emissions)/1e9, "5.2e")) + " Mt of CO2e")
            output_em.grid(row=0, column=1, sticky='W', pady=10)
        counter = 0
        dispSN= tk.Listbox(self, height=16)
        scrollbar = tk.Scrollbar(self)
        for key in total_mat_emissions_per_year:
            dispSN.insert(counter, key+": \n \t ")
            dispSN.insert(counter+1, "\t"+str(format(sum(total_mat_emissions_per_year[key].mean(1)), "5.2e"))+"kg of CO2e \n")
            counter += 2
        scrollbar.grid(row=3, column=0, rowspan=7, sticky='NSE', pady=10)
        dispSN.grid(row=2, column=0, rowspan=8, sticky='E', pady=10)

        dispSN.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=dispSN.yview)
        allModes = TransportModes + FreightModes + InfraModes
        names = []
        for obj in allModes:
            names.append(obj.name)
        fillerCount=10 #3
        selMode = tk.Label(self, text="Graph vehicle mode emissions: ")
        selMode.grid(row=2, column=1, sticky='W', pady=10)
        self.variable = tk.StringVar(self)
        self.variable.set(names[0])
        self.mode_list = tk.OptionMenu(self, self.variable, *names)
        self.mode_list.grid(row= 2, column=2,columnspan=2, sticky='W', pady=1)
        get_graphA = tk.Button(self, text="Graph Emissions", command=lambda: allModes[names.index(self.variable.get())].graph_emissions(4))
        get_graphA.grid(row=2, column=3, sticky='E', pady=10)
        get_graphB = tk.Button(self, text="Graph Amounts",
                               command=lambda: allModes[names.index(self.variable.get())].graph_amounts(5))
        get_graphB.grid(row=2, column=4, sticky='E', pady=10)

        button = tk.Button(self, text="Return Scenario One",
                           command=lambda: controller.show_frame("ScenOne"))
        button.grid(row=len(names)+fillerCount, column=0, sticky='E', pady=10)
        button = tk.Button(self, text="pie chart of Vehicle Emissions",
                           command=lambda: plot_pie(len(names)+12, total_vehicle_emissions_per_year,
                                                         "Pie Chart of Vehicle Emissions"))
        button.grid(row=len(names) + fillerCount, column=1, sticky='E', pady=10)
        graph_mat = tk.Button(self, text="pie chart of material Amounts",
                           command=lambda: plot_pie(len(names) + 13, tot_mat_amounts_per_year,
                                                         "Pie Chart of Material Amounts"))
        graph_mat.grid(row=len(names) + fillerCount, column=2, sticky='E', pady=10)
        graph_matE = tk.Button(self, text="pie chart of material Emissions",
                              command=lambda: plot_pie(len(names) + 14, total_mat_emissions_per_year,
                                                            "Pie Chart of Material Emissions"))
        graph_matE.grid(row=len(names) + fillerCount, column=4, sticky='E', pady=10)
        if len(emissions)>1:
            if np.mean(emissions)>1000:
                plot_figures(np.mean(emissions))



def plot_figures(meanEmVal):
    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.show(block=False)
    plotlist = []
    counter = 0
    bars = []
    allModes = TransportModes + FreightModes + InfraModes
    names = []
    for obj in allModes:
        matEmissionsvalue = {}
        matEmissionsSD = {}
        for key in total_mat_emissions_per_year:
            if key in obj.mat_emissions_per_year:
                matEmissionsvalue[key] = sum(list(obj.mat_emissions_per_year[key].mean(1)))
                matEmissionsSD[key] = sum(list(obj.mat_emissions_per_year[key].std(1)))
            else:
                matEmissionsvalue[key] = 0
                matEmissionsSD[key] = 0

        names.append(obj.name)
        if counter == 0:
            plotlist.append(plt.bar(range(len(matEmissionsvalue)), list(matEmissionsvalue.values()),
                                    yerr=list(matEmissionsSD.values()),
                                    color=pal[counter], ecolor=pal[counter], capsize=5, alpha=0.9, align='center'))
            bars = list(matEmissionsvalue.values())
        else:
            plotlist.append(plt.bar(range(len(matEmissionsvalue)), list(matEmissionsvalue.values()),
                                    yerr=list(matEmissionsSD.values()),
                                    color=pal[counter], alpha=0.9, bottom=bars, ecolor=pal[counter], capsize=5,
                                    align='center'))
            for i in range(len(matEmissionsvalue)):
                bars[i] = bars[i] + list(matEmissionsvalue.values())[i]
        counter += 1
        plt.xticks(range(len(matEmissionsvalue)), list(matEmissionsvalue.keys()), rotation=60)
    plt.legend(plotlist, names)
    plt.ylabel("Kg of CO$_{2e}$")
    # figure 2 for driving emissions
    plt.figure(2)
    # plt.clf()
    plt.ion()
    driv_emissionsplot={}
    driv_emissionsplotSD={}
    # one can use the total emissions value passed into this method to graph all driving emissions on the same graph
    palED = sns.color_palette("Blues_r")
    for key in total_driving_emissions_per_year:
        driv_emissionsplot[key] = sum(list(total_driving_emissions_per_year[key].mean(1)))
        driv_emissionsplotSD[key] = sum(list(total_driving_emissions_per_year[key].std(1)))
    plt.bar(np.arange(len(driv_emissionsplot)), list(driv_emissionsplot.values()),
            yerr=list(driv_emissionsplotSD.values()),
            color=palED[1], ecolor=palED[0], capsize=5, alpha=0.9, align='center')

    plt.xticks(np.arange(len(driv_emissionsplot)), list(driv_emissionsplot.keys()), rotation=60)
    plt.title("CO$_{2e}$ emissions from Operation")
    plt.ylabel("Kg of CO$_{2e}$")
    plt.figure(3)
    plt.clf()
    plt.ion()
    pals = sns.color_palette("hls", 15)
    num=0
    for key in total_mat_emissions_per_year:
        plt.plot([x+2020 for x in range(totalYears)], total_mat_emissions_per_year[key].mean(1), color=pals[num])
        num += 1
    plt.legend(total_mat_emissions_per_year.keys())
    plt.ylabel("Kg of CO$_{2e}$")
    plt.xticks([x+2020 for x in range(0,totalYears, 2)])


def plot_pie(num, plot_dict, title):
    plt.figure(num)
    plt.clf()
    plt.show(block=False)
    plt.ion()
    dictvals={}
    for key in plot_dict:
        dictvals[key] = sum(plot_dict[key].mean(0))
    plt.pie(dictvals.values(),colors=pal)
    plt.legend(list(plot_dict.keys()))
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'})


class Vehicle:
    def __init__(self, vehicle_name, materials_dict={}, standard_dev=0, occupancy=0, tonnes=0, life_distance=0,
                 defaultPer = 0, co2perkm = .22, techAdv=0):
        self.name = vehicle_name
        self.defaultPer = defaultPer
        self.standard_dev = standard_dev
        self.techAdv= techAdv
        self.materialList = {}
        self.mat_emissions = {}
        self.veh_emissions = 0
        self.mat_amounts = {}
        self.mat_amounts_per_year = {}
        self.mat_emissions_per_year = {}
        self.driv_emissions = 0
        self.occupancy = occupancy  # number of people
        self.tonnes = tonnes  # kg
        self.co2_perkm = co2perkm
        self.life_distance = life_distance  # km
        self.add_materials(materials_dict, standard_dev)
        for key in self.materialList:
            self.mat_amounts_per_year[key] = np.zeros([totalYears,sobolCount*7])
            self.mat_emissions_per_year[key] = np.zeros([totalYears,sobolCount*7])

    def add_materials(self, materials_dict, standard_dev):
        for Key, Value in materials_dict.items():
            if Key not in self.materialList.keys():
                self.materialList[Key] = Material(Key, Value, standard_dev)

    def get_emissions(self, pkm, percentsPkm, gridIntensity, perElectric, year, j, tech):
        percent = percentsPkm[self.name]
        total_co2 = 0
        self.mat_emissions = {}
        occorton = 0
        if self.occupancy != 0:
            occorton = self.occupancy
        else:
            occorton = self.tonnes
        vkm = ((pkm * (percent / 100)) / occorton)
        for key in self.materialList:
            if key == 'Battery':
                self.mat_amounts[key] = vkm / self.life_distance * (perElectric/100) * np.random.normal(self.materialList[key].amount, ((self.materialList[key].amount * self.materialList[key].sD)-self.materialList[key].amount))
                self.mat_emissions[key] = [(self.mat_amounts[key] * np.random.normal(emissions_per_kgmat[key][0],
                                                                                     (emissions_per_kgmat[key][0]* emissions_per_kgmat[key][1])-emissions_per_kgmat[key][0])),
                                           (math.sqrt((self.standard_dev - 1) ** 2 + (
                                                       emissions_per_kgmat[key][1] - 1) ** 2))]
                total_co2 += self.mat_emissions[key][0]
                self.mat_amounts_per_year[key][year][j] = self.mat_amounts[key]
                self.mat_emissions_per_year[key][year][j] = self.mat_emissions[key][0]
            else:
                self.mat_amounts[key] = vkm / self.life_distance * np.random.normal(self.materialList[key].amount, ((self.materialList[key].amount * self.materialList[key].sD)-self.materialList[key].amount))
                self.mat_emissions[key] = [(self.mat_amounts[key] * np.random.normal(emissions_per_kgmat[key][0],
                                                                                     (emissions_per_kgmat[key][0]* emissions_per_kgmat[key][1])-emissions_per_kgmat[key][0])),
                                           (math.sqrt((self.standard_dev-1)**2+(emissions_per_kgmat[key][1]-1)**2))]
                total_co2 += self.mat_emissions[key][0]
                self.mat_amounts_per_year[key][year][j] = self.mat_amounts[key]
                self.mat_emissions_per_year[key][year][j] = self.mat_emissions[key][0]
        self.driv_emissions = (vkm * (1-(perElectric/100)) * self.co2_perkm*(1-(self.techAdv*tech))) + \
                              (pkm * percent / 100 * (perElectric/100) / occorton * .2 * gridIntensity)
        total_co2 += self.driv_emissions
        self.veh_emissions = total_co2
        return total_co2, self.mat_emissions, self.driv_emissions

    def graph_emissions(self, val):
        plt.figure(val)
        plt.clf()
        plt.ion()
        plt.show(block=False)
        matEmissionsvalue = {}
        matEmissionsSD = {}
        for key in self.mat_emissions_per_year:
            matEmissionsvalue[key] = sum(list(self.mat_emissions_per_year[key].mean(1)))
            matEmissionsSD[key] = sum(list(self.mat_emissions_per_year[key].std(1)))
        palED = sns.color_palette("Blues_r")
        print(list(matEmissionsvalue.values()))
        plt.bar(range(len(matEmissionsvalue)), list(matEmissionsvalue.values()),
                yerr=list(matEmissionsSD.values()), color =palED[1],
                ecolor=palED[0], capsize=5, alpha=1, align='center')
        plt.ylabel("Kg of CO$_{2e}$")
        plt.title("Emissions due to " + self.name)
        plt.xticks(range(len(matEmissionsvalue)), list(matEmissionsvalue.keys()), rotation=60)


    def graph_amounts(self, val):
        plt.figure(val)
        plt.clf()
        plt.ion()
        plt.show(block=False)
        matAmountvalue = {}
        matAmountSD = {}
        for key in self.mat_amounts_per_year:
            matAmountvalue[key] = sum(list(self.mat_amounts_per_year[key].mean(1)))
            matAmountSD[key] = sum(list(self.mat_amounts_per_year[key].std(1)))
        palED = sns.color_palette("Blues_r")
        plt.bar(range(len(matAmountvalue)), list(matAmountvalue.values()),
                yerr=list(matAmountSD.values()),  color =palED[1],
                ecolor=palED[0], capsize=5, alpha=1, align='center')
        plt.ylabel("Kg of Materials")
        plt.title("Amount of Materials due to " + self.name)
        plt.xticks(range(len(matAmountvalue)), list(matAmountvalue.keys()), rotation=60)
        plt.figure(val + 1)
        plt.clf()
        plt.ion()
        num = 0
        for key in self.mat_amounts_per_year:
            plt.errorbar([x + 2020 for x in range(totalYears)], list(self.mat_amounts_per_year[key].mean(1)),
                         yerr=list(self.mat_amounts_per_year[key].std(1)), color=pal[num])
            num += 1
        plt.legend(self.mat_amounts_per_year.keys())
        plt.title("Total Material Amounts per Year for " + self.name)
        plt.xticks([x + 2020 for x in range(0, totalYears, 2)])
        plt.ylabel('Kg')


class Infrastructure:
    def __init__(self, infra_name, materials_dict={}, standard_dev=0, life_distance=0, unit='#', defaultKm=0):
        self.name = infra_name
        self.defaultKm = defaultKm
        self.unit = unit
        self.mat_emissions = {}
        self.mat_amounts = {}
        self.mat_amounts_per_year = {}
        self.mat_emissions_per_year = {}
        self.standard_dev = int(standard_dev)
        self.materialList = {}
        self.life_distance = life_distance  # km
        self.add_materials(materials_dict, int(standard_dev))
        for key in self.materialList:
            self.mat_amounts_per_year[key] = np.zeros([totalYears, sobolCount*7])
            self.mat_emissions_per_year[key] = np.zeros([totalYears, sobolCount*7])

    def add_materials(self, materials_dict, standard_dev):

        for Key, Value in materials_dict.items():
            if Key not in self.materialList.keys():
                self.materialList[Key] = Material(Key, Value, standard_dev)

    def get_emissions(self, km, gridIntensity, year, j):
        total_co2 = 0
        self.mat_emissions = {}
        for key in self.materialList:
            self.mat_amounts[key] = km * np.random.normal(self.materialList[key].amount, (int(self.materialList[key].amount * self.materialList[key].sD)-self.materialList[key].amount))
            self.mat_emissions[key] = [self.mat_amounts[key] * np.random.normal(emissions_per_kgmat[key][0],(emissions_per_kgmat[key][0]*emissions_per_kgmat[key][1])-emissions_per_kgmat[key][0]),
                                       (math.sqrt((self.standard_dev-1)**2+(emissions_per_kgmat[key][1]-1)**2))]
            total_co2 += self.mat_emissions[key][0]
            self.mat_amounts_per_year[key][year][j] = self.mat_amounts[key]
            self.mat_emissions_per_year[key][year][j] = self.mat_emissions[key][0]
        return total_co2, self.mat_emissions

    def graph_emissions(self, val):
        plt.figure(val)
        plt.clf()
        plt.ion()
        plt.show(block=False)
        matEmissionsvalue = {}
        matEmissionsSD = {}
        for key in self.mat_emissions_per_year:
            matEmissionsvalue[key] = sum(list(self.mat_emissions_per_year[key].mean(1)))
            matEmissionsSD[key] = sum(list(self.mat_emissions_per_year[key].std(1)))
        palED = sns.color_palette("Blues_r")
        print(list(matEmissionsvalue.values()))
        plt.bar(range(len(matEmissionsvalue)), list(matEmissionsvalue.values()),
                yerr=list(matEmissionsSD.values()), color =palED[1],
                ecolor=palED[0], capsize=5, alpha=1, align='center')
        plt.ylabel("Kg of CO$_{2e}$")
        plt.title("Emissions due to " + self.name)
        plt.xticks(range(len(matEmissionsvalue)), list(matEmissionsvalue.keys()), rotation=60)

    def graph_amounts(self, val):
        plt.figure(val)
        plt.clf()
        plt.ion()
        plt.show(block=False)
        matAmountvalue = {}
        matAmountSD = {}
        for key in self.mat_amounts_per_year:
            matAmountvalue[key] = sum(list(self.mat_amounts_per_year[key].mean(1)))
            matAmountSD[key] = sum(list(self.mat_amounts_per_year[key].std(1)))
        print(list(matAmountvalue.values()))
        print(list(matAmountSD.values()))
        palED = sns.color_palette("Blues_r")
        plt.bar(range(len(matAmountvalue)), list(matAmountvalue.values()),
                yerr=list(matAmountSD.values()), color=palED[1],
                ecolor=palED[0], capsize=5, alpha=1, align='center')
        plt.ylabel("Kg of Materials")
        plt.title("Amount of Materials due to " + self.name)
        plt.xticks(range(len(matAmountvalue)), list(matAmountvalue.keys()), rotation=60)
        plt.figure(val + 1)
        plt.clf()
        plt.ion()
        num = 0
        for key in self.mat_amounts_per_year:
            plt.errorbar([x + 2020 for x in range(totalYears)], list(self.mat_amounts_per_year[key].mean(1)),
                         yerr=list(self.mat_amounts_per_year[key].std(1)), color=pal[num])
            num += 1
        plt.legend(self.mat_amounts_per_year.keys())
        plt.title("Total Material Amounts per Year for " + self.name)
        plt.xticks([x + 2020 for x in range(0, totalYears, 2)])
        plt.ylabel('Kg')


class Material:
    def __init__(self, material_name, amount, sD):
        self.name = material_name
        self.amount = amount
        self.sD = sD

    def __str__(self):
        return 'Material = ' + self.name + '\n \tAmount = ' + str(self.amount) + '\n \tStandard Deviation = ' + str(
            self.sD)


# country class can be expanded in the future and could include the various grid emissions factors for each country
class Country:
    def __init__(self, country_name):
        self.name = country_name

    def __str__(self):
        return self.name


emissions_per_kgmat = {'Steel': [1.82, 1.203], 'aluminium': [7.916, 1.203], 'Iron': [2.03, 1.219], 'Copper': [2.303, 1.203],
                       'Rubber': [2.85, 1.219], 'Plastic': [4.05, 1.203],'Glass': [.95, 1.203],
                       'Textiles': [2.15, 1.203], 'Solid Rock': [.00323, 1.222], 'Fly Ash': [.041, 1.203], 'Sand and Gravel': [0.00224 , 1.222], 'Asphalt': [.076, 1.203], 'Cement': [.949, 1.222], 'Battery': [14.45, 1.219],
                       'Wood': [.08, 1.222]}
LDV = Vehicle('Light Duty Vehicle',
              {'Steel': 980, 'aluminium': 137, 'Iron': 156, 'Copper': 28,
               'Rubber': 81, 'Plastic': 21, 'Glass': 144, 'Textiles': 46, 'Battery': 420}, 1.219, occupancy=2.58,
              life_distance=200000, defaultPer=13.9, co2perkm=.275, techAdv=.1)
PassTrain = Vehicle('Passenger Train',
                    {'Steel': 114000, 'aluminium': 10000,
                     'Copper': 7000, 'Plastic': 13000, 'Glass': 7230, 'Wood': 6675}, 1.302, occupancy=270,
                    life_distance=6000000, defaultPer=.01, co2perkm=8.873, techAdv=.0)
Bus = Vehicle('Bus',
              {'Steel': 6630, 'aluminium': 654, 'Iron': 0, 'Copper': 93,
               'Rubber': 1392, 'Plastic': 289, 'Glass': 327, 'Textiles': 0, 'Battery': 700}, 1.260, occupancy=45,
              life_distance=950000, defaultPer=17.5, co2perkm=.420, techAdv=.1)
Minibus = Vehicle('Minibus',
                 {'Steel': 1406, 'aluminium': 196, 'Iron': 224, 'Copper': 40,
                  'Rubber': 116, 'Plastic': 30, 'Glass': 206, 'Textiles': 66, 'Battery': 602}, 1.688, occupancy=12,
                 life_distance=250000, defaultPer=59.4, co2perkm=.319, techAdv=.1)
Motorcycle = Vehicle('Motorcycle',
              {'Steel': 55.29, 'aluminium': 15, 'Iron': 0, 'Copper': 1.352,
               'Rubber': 2.95, 'Plastic': 125.05, 'Glass': 0, 'Textiles': 0,'Battery':32}, 1.684, occupancy=1.73,
                     life_distance=100000, defaultPer=9.19, co2perkm=.041, techAdv=.0)
LGV = Vehicle('Light Goods Vehicle',
              {'Steel': 1257, 'aluminium': 175, 'Iron': 201, 'Copper': 35.9,
               'Rubber': 103.9, 'Plastic': 37.2, 'Glass': 184.8, 'Textiles': 59, 'Battery': 538}, 1.688, tonnes=.97,
              life_distance=200000, defaultPer=19.1, co2perkm=.319, techAdv=.10)
Truck = Vehicle('Truck',
                {'Steel': 2276, 'aluminium': 215, 'Iron': 3080, 'Copper': 50,
                 'Rubber': 375, 'Plastic': 330, 'Glass': 45, 'Textiles': 35, 'Battery': 2100}, 1.260, tonnes=5.7, occupancy=0,
                life_distance=750000, defaultPer=76.3, co2perkm=1.129, techAdv=.1)
FreightTrain = Vehicle('Freight Train',
                       {'Steel': 53410, 'aluminium': 5550,'Copper': 6810,'Glass': 2350,'Plastic': 6340,
                         'Wood': 456.61}, 1.651,
                       tonnes=60, life_distance=9600000, defaultPer=4.6, co2perkm=8.873, techAdv=.001)

Road = Infrastructure('Road',
                      {'Sand and Gravel': 12112800, 'Cement': 126420, 'Solid Rock': 2883600, 'Fly Ash': 4609920,
                        'Asphalt': 4306560}, 1.381, defaultKm=300)

TrainTrack = Infrastructure('TrainTrack', {'Solid Rock': 147435435 , 'Sand and Gravel': 667081, 'Cement': 529087,
                                           'Iron': 501863, 'Wood': 26609}, 1.246, defaultKm=50)
TransportModes = [LDV, PassTrain, Bus, Minibus, Motorcycle]
FreightModes = [LGV, Truck, FreightTrain]
InfraModes = [Road, TrainTrack]
allModes = TransportModes + FreightModes + InfraModes

Uganda = Country('Uganda')
country_list[Uganda.name] = Uganda
if __name__ == "__main__":
    app = FrameApp()
    app.mainloop()
