import argparse
import yaml
#import multiprocessing as mp
import modules.class_fourteen_matricesS
import modules.class_fourteen_matricesP
import modules.class_fourteen_matrices_spinS
import modules.class_fourteen_matrices_spinP
import modules.current_volt
import generate_matrices.system_def
import generate_matrices.system_def_CP2K
import generate_matrices.system_def_CP2K_h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--build", \
                        help="Building system (empirical)", nargs="+")
    parser.add_argument("-bCP2K", "--buildCP2K",\
                        help="Building system with CP2K matrices", nargs="+")
    parser.add_argument("-cs", "--config_sequential",\
                        help="Config file sequential calculation", nargs="+")
    parser.add_argument("-cp", "--config_parallel",\
                        help="Config file parallel calculation", nargs="+")
    parser.add_argument("-scs", "--spincfg_sequential",\
                        help="Config file with spin. Sequential", nargs="+")
    parser.add_argument("-scsf", "--spincfg_sequential_f",\
                        help=("Config file with spin. Sequential. Fortran\
                               support"), nargs="+")
    parser.add_argument("-scp", "--spincfg_parallel",\
                        help="Config file with spin. Parallel", nargs="+")
    parser.add_argument("-scpf", "--spincfg_parallel_f",\
                        help=("Config file with spin. Parallel. Fortran\
                               support"), nargs="+")
    parser.add_argument("-cbscs", "--coulblockspin_sequential",\
                        help="Config file with spin in Coulomb \
                        Blockade regime. Sequential")
    parser.add_argument("-cbscp", "--coulblockspin_parallel",\
                        help="Config file with spin in Coulomb \
                        Blockade regime. Parallel")
    parser.add_argument("-ivc", "--iv_characteristics",\
                        help="Calculate I-V characteristics", nargs="+")
    parser.add_argument("-ivcb", "--iv_characteristics_cb",\
                        help="Calculate I-V characteristics from \
                        Coulomb Blockade method")
    parser.add_argument("-p", "--printing", help="Printing of files", nargs="+")

    args = parser.parse_args()
    
    # Build 14 matrices empirically
    if args.build:
        for i in args.build:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            generate_matrices.system_def.build_and_dump(cfg)

    #elif args.buildCP2K:
    #    for i in args.buildCP2K:
    #        with open(i,"r") as yamlfile:
    #            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    #        generate_matrices.system_def_CP2K.build_and_dump(cfg)

    # Build 14 or 16 matrices from a CP2K *.ao file
    elif args.buildCP2K:
        for i in args.buildCP2K:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            generate_matrices.system_def_CP2K_h5py.build_and_dump(cfg)

    # Calculate DOS and T(E). Spin Unpolarized systems. Single Particle (1 Node)
    elif args.config_sequential:
        for i in args.config_sequential:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.class_fourteen_matricesS.fourteen_matrices(cfg)
            system.NEGF()

    # Calculate DOS and T(E). Spin unpolarized systems. Single Particle (N Nodes)
    elif args.config_parallel:
        for i in args.config_parallel:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.class_fourteen_matricesP.fourteen_matrices(cfg)
            system.NEGF()
    
    # Calculate DOS and T(E). Spin Polarized systems. Single Particle (1 Node)
    elif args.spincfg_sequential:
        for i in args.spincfg_sequential:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.class_fourteen_matrices_spinS.fourteen_matrices_spin(cfg)
            system.NEGF()

    # Calculate DOS and T(E). Spin Polarized systems. Single Particle (1 Node)
    # Use fortran Routines for accelerated matrix handling (Linear Algebra)
    elif args.spincfg_sequential_f:
        for i in args.spincfg_sequential_f:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.class_fourteen_matrices_spinS.FET_DOST_Spin_fort(cfg)
            system.NEGF()

    # Calculate DOS and T(E). Spin Polarized systems. Single Particle (N Nodes)
    elif args.spincfg_parallel:
        for i in args.spincfg_parallel:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system = modules.class_fourteen_matrices_spinP.fourteen_matrices_spin(cfg)
            system.NEGF()

    # Calculate DOS and T(E). Spin Polarized systems. Single Particle (N Nodes)
    # Use fortran Routines for accelerated matrix handling (Linear Algebra)
    elif args.spincfg_parallel_f:
        for i in args.spincfg_parallel_f:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system = modules.class_fourteen_matrices_spinP.FET_DOST_Spin_fort(cfg)
            system.NEGF()
    
    # Calculate DOS and T(E). Spin Polarized systems. Coulomb Blockade (1 Node)
    elif args.coulblockspin_sequential:
        with open(args.coulblockspin_sequential,"r") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        system  = modules.class_fourteen_matrices_spinS.fourteen_matrices_spin(cfg)
        system.NEGF_CB()
    
    # Calculate DOS and T(E). Spin Polarized systems. Coulomb Blockade (N Nodes)
    elif args.coulblockspin_parallel:
        with open(args.coulblockspin_parallel,"r") as yamlfile:
            cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        system  = modules.class_fourteen_matrices_spinP.fourteen_matrices_spin(cfg)
        system.NEGF_CB()
    
    elif args.iv_characteristics:
        for i in args.iv_characteristics:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.current_volt.iv_characteristics(cfg)
            system.iv_curves()
    
    elif args.iv_characteristics_cb:
            with open(args.iv_characteristics_cb,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.current_volt.iv_characteristics(cfg)
            system.iv_curves_cb()
    
    elif args.printing:
        for i in args.printing:
            with open(i,"r") as yamlfile:
                cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
            system  = modules.class_fourteen_matrices_spinS.fourteen_matrices_spin(cfg)
            system.plot()
