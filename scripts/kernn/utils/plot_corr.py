import numpy as np
import matplotlib.pyplot as plt

#define constants
evtokcal = 23.060541945

def plot_corr(e_true, e_pred, F, forces, model_path):

    # definitions for the axes
    left, width = 0.15, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax1 = plt.axes(rect_scatter)
    ax1.tick_params(direction='in', top=True, right=True)
    ax0 = plt.axes(rect_histx)
    ax0.tick_params(direction='in', labelbottom=False)

    ax1.set_aspect('equal', adjustable='box')
    # Fontsize
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #plot section
    #global params
    plt.rcParams['lines.markersize'] = 12
    #plt.rcParams['axes.linewidth'] = 20000
    for axis in ['top','bottom','left','right']:
      ax0.spines[axis].set_linewidth(2)
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)



    ax0.axhline(0.0, linestyle='--', color='red')


    #ax1.set_ylim(-240, -140)
    #ax1.set_xlim(-240, -140)

    ax0.tick_params(width=2, length=10, labelsize=15)
    ax0.tick_params(which='minor',width=1, length=5, color='black', labelsize=15)
    ax1.tick_params(width=2, length=10, labelsize=15)
    ax1.tick_params(which='minor',width=1, length=5, color='black', labelsize=15)


    ax0.set_ylabel('$\\Delta E$ (kcal/mol)', fontsize=18)
    ax1.set_ylabel('$E_{\\rm KerNN}$ (kcal/mol)', fontsize=18)
    ax1.set_xlabel('$E_{\\rm Ref.}$ (kcal/mol)', fontsize=18)
    


    ax0.scatter(e_true, e_pred - e_true)
    ax1.scatter(e_true, e_pred)
    #ax1.plot([-240, -140], [-240, -140], "r--")
    plt.savefig(model_path + 'plot_corr.png',bbox_inches='tight', dpi=250)

    plt.show()


    print("AVERAGED ERRORS")
    mae = np.mean(np.abs(e_true - e_pred))
    print("MAE(E): ", mae)
    
    rmse = np.sqrt(np.sum((e_true - e_pred)**2)/len(e_true))
    print("RMSE(E): ", rmse)
    
    from sklearn.metrics import mean_squared_error, r2_score
    print("R2 (E):" , r2_score(e_true, e_pred))

    #check the accuracy of the forces
    # function to calculate the mae and mse with numpy (no tensorstuff)
    def calculate_mae_mse(val1, val2):
        delta = val1 - val2
        mse = np.mean(delta ** 2)
        mae = np.mean(np.abs(delta))
        return mae, mse

    print("MAE(F): ", calculate_mae_mse(F, forces)[0])
    print("RMSE(F): ", np.sqrt(calculate_mae_mse(F, forces)[1]))


    np.savetxt(model_path + "averaged_errors.dat", [mae, rmse, r2_score(e_true, e_pred), calculate_mae_mse(F, forces)[0], np.sqrt(calculate_mae_mse(F, forces)[1])])
