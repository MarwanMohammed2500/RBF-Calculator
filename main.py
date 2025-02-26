from RBF import RBF

def main():
    rbf = RBF()
    rbf.calc_r()
    rbf.calc_phi()
    rbf.show_table()
    rbf.plot_x()
    rbf.plot_phi()
    

if __name__ == "__main__":
    main()