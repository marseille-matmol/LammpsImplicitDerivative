LAMMPS (2 Aug 2023)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task

        # Reset simulation state
        clear
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task
        # Configure how atoms are mapped to store and access
        atom_modify map array sort 0 0.0
        # metal units: Angstroms, eV, ps, etc.
        units metal
        

        clear
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (src/comm.cpp:98)
  using 1 OpenMP thread(s) per MPI task

        atom_modify map array sort 0 0.0
        boundary p p p

        # Initialize simulation
        units metal
        lattice bcc 3.18427 origin 0.01 0.01 0.01
Lattice spacing in x,y,z = 3.18427 3.18427 3.18427

        # create a block of atoms
        region C block 0 2 0 2 0 2 units lattice
        create_box 1 C
Created orthogonal box = (0 0 0) to (6.36854 6.36854 6.36854)
  1 by 1 by 1 MPI processor grid
        create_atoms 1 region C
Created 16 atoms
  using lattice units in orthogonal box = (0 0 0) to (6.36854 6.36854 6.36854)
  create_atoms CPU = 0.000 seconds

        mass * 184.0
        
run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
WARNING: Communication cutoff is 0.0. No ghost atoms will be generated. Atoms may get lost. (src/comm_brick.cpp:210)
Per MPI rank memory allocation (min/avg/max) = 2.301 | 2.301 | 2.301 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         0   0              0              0              0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 16 atoms

104.4% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 0
Ave neighs/atom = 0
Neighbor list builds = 0
Dangerous builds = 0

        pair_style snap

        # Take coefficients from files W.snapcoeff and W.snapparam
        pair_coeff * * ../../lammps_implicit_der/data_files/W_NEW.snapcoeff ../../lammps_implicit_der/data_files/W_NEW.snapparam W
SNAP Element = W, Radius 0.5, Weight 1
Reading potential file ../../lammps_implicit_der/data_files/W_NEW.snapparam with DATE: 2017-02-20
SNAP keyword rcutfac 5.0
SNAP keyword twojmax 8
SNAP keyword rfac0 0.99363
SNAP keyword rmin0 0
SNAP keyword bzeroflag 1
SNAP keyword quadraticflag 0
SNAP keyword bnormflag 0
        run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7
  ghost atom cutoff = 7
  binsize = 3.5, bins = 2 2 2
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair snap, perpetual
      attributes: full, newton on
      pair build: full/bin/atomonly
      stencil: full/bin/3d
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.557 | 3.557 | 3.557 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         0   0             -89.06089       0             -89.06089      -18.044       
Loop time of 8.34e-07 on 1 procs for 0 steps with 16 atoms

119.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
        
reset_timestep 0

        min_style cg
        minimize 0 1e-08 1000 1000
Per MPI rank memory allocation (min/avg/max) = 4.682 | 4.682 | 4.682 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         0   0             -89.06089       0             -89.06089      -18.044       
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 0.0123359 on 1 procs for 1 steps with 16 atoms

97.8% CPU use with 1 MPI tasks x 1 OpenMP threads

Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = 
      -89.060889667491   -89.060889667491   -89.060889667491
  Force two-norm initial, final = 3.9534553e-14 4.305488e-14
  Force max component initial, final = 1.0214052e-14 1.1768364e-14
  Final line search alpha, max atom move = 1 1.1768364e-14
  Iterations, force evaluations = 1 2

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.012287   | 0.012287   | 0.012287   |   0.0 | 99.60
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 1.1291e-05 | 1.1291e-05 | 1.1291e-05 |   0.0 |  0.09
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 3.767e-05  |            |       |  0.31

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
        
WARNING: lammps_gather: unknown compute id (src/library.cpp:3787)
WARNING: lammps_gather: unknown compute id (src/library.cpp:3787)

            # descriptors
            compute D all sna/atom 5.0 0.99363 8 0.5 1.0

            # derivatives of descriptors
            compute dD all snad/atom 5.0 0.99363 8 0.5 1.0

            # potential energy per atom
            compute E all pe/atom

            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7
  ghost atom cutoff = 7
  binsize = 3.5, bins = 2 2 2
  3 neighbor lists, perpetual/occasional/extra = 1 2 0
  (1) pair snap, perpetual
      attributes: full, newton on
      pair build: full/bin/atomonly
      stencil: full/bin/3d
      bin: standard
  (2) compute sna/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
  (3) compute snad/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
Per MPI rank memory allocation (min/avg/max) = 4.693 | 4.693 | 4.693 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

133.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            
run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 4.693 | 4.693 | 4.693 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 7.91e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
reset_timestep 0

        min_style cg
        minimize 0 1e-08 1000 1000
Per MPI rank memory allocation (min/avg/max) = 34.86 | 34.86 | 34.86 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         0   0             -89.06089       0             -89.06089      -18.044       
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 0.0122797 on 1 procs for 1 steps with 16 atoms

98.7% CPU use with 1 MPI tasks x 1 OpenMP threads

Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final = 
      -89.060889667491   -89.060889667491   -89.060889667491
  Force two-norm initial, final = 4.305488e-14 3.6520053e-14
  Force max component initial, final = 1.1768364e-14 1.110223e-14
  Final line search alpha, max atom move = 1 1.110223e-14
  Iterations, force evaluations = 1 2

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.012233   | 0.012233   | 0.012233   |   0.0 | 99.62
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 8.457e-06  | 8.457e-06  | 8.457e-06  |   0.0 |  0.07
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 3.834e-05  |            |       |  0.31

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
        
run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 33.73 | 33.73 | 33.73 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 9.58e-07 on 1 procs for 0 steps with 16 atoms

104.4% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0

            # descriptors
            compute virial all snav/atom 5.0 0.99363 8 0.5 1.0
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7
  ghost atom cutoff = 7
  binsize = 3.5, bins = 2 2 2
  4 neighbor lists, perpetual/occasional/extra = 1 3 0
  (1) pair snap, perpetual
      attributes: full, newton on
      pair build: full/bin/atomonly
      stencil: full/bin/3d
      bin: standard
  (2) compute sna/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
  (3) compute snad/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
  (4) compute snav/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
Per MPI rank memory allocation (min/avg/max) = 33.8 | 33.8 | 33.8 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

133.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            
run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 75.82 | 75.82 | 75.82 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 7.08e-07 on 1 procs for 0 steps with 16 atoms

141.2% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0

            # descriptors
            compute virial all snav/atom 5.0 0.99363 8 0.5 1.0
ERROR: Reuse of compute ID 'virial' (src/modify.cpp:1240)
Last command:             compute virial all snav/atom 5.0 0.99363 8 0.5 1.0

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
            change_box all x final 0.0 6.050113 y final 0.0 6.050113 z final 0.0 6.050113 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.050113 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.050113 6.050113 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.050113 6.050113 6.050113) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.050113 6.050113 6.050113) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.050113 6.050113 6.050113) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.050113 6.050113 6.050113) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Neighbor list info ...
  update: every = 1 steps, delay = 0 steps, check = yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7
  ghost atom cutoff = 7
  binsize = 3.5, bins = 2 2 2
  4 neighbor lists, perpetual/occasional/extra = 1 3 0
  (1) pair snap, perpetual
      attributes: full, newton on
      pair build: full/bin/atomonly
      stencil: full/bin/3d
      bin: standard
  (2) compute sna/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
  (3) compute snad/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
  (4) compute snav/atom, occasional, copy from (1)
      attributes: full, newton on
      pair build: copy
      stencil: none
      bin: none
Per MPI rank memory allocation (min/avg/max) = 75.55 | 75.55 | 75.55 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -80.90675       0             -80.90675       766512.41    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

149.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.050113 6.050113 6.050113) with tilt (0 0 0)
            change_box all x final 0.0 6.060727233333334 y final 0.0 6.060727233333334 z final 0.0 6.060727233333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.0607272 6.050113 6.050113) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.050113) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.0607272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.0607272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.0607272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.0607272) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -81.45468       0             -81.45468       737253.36    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.0607272 6.0607272 6.0607272) with tilt (0 0 0)
            change_box all x final 0.0 6.071341466666667 y final 0.0 6.071341466666667 z final 0.0 6.071341466666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.0713415 6.0607272 6.0607272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0607272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0713415) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0713415) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0713415) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0713415) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -81.9832        0             -81.9832        708172.86    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

184.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.0713415 6.0713415 6.0713415) with tilt (0 0 0)
            change_box all x final 0.0 6.0819557 y final 0.0 6.0819557 z final 0.0 6.0819557 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.0819557 6.0713415 6.0713415) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0713415) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0819557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0819557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0819557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0819557) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -82.492297      0             -82.492297      679279.76    
Loop time of 6.66e-07 on 1 procs for 0 steps with 16 atoms

300.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.0819557 6.0819557 6.0819557) with tilt (0 0 0)
            change_box all x final 0.0 6.092569933333333 y final 0.0 6.092569933333333 z final 0.0 6.092569933333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.0925699 6.0819557 6.0819557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0819557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0925699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0925699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0925699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0925699) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -82.981968      0             -82.981968      650582.71    
Loop time of 5.84e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.0925699 6.0925699 6.0925699) with tilt (0 0 0)
            change_box all x final 0.0 6.103184166666667 y final 0.0 6.103184166666667 z final 0.0 6.103184166666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1031842 6.0925699 6.0925699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.0925699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.1031842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.1031842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.1031842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.1031842) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -83.452215      0             -83.452215      622090.19    
Loop time of 5.41e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1031842 6.1031842 6.1031842) with tilt (0 0 0)
            change_box all x final 0.0 6.1137984 y final 0.0 6.1137984 z final 0.0 6.1137984 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1137984 6.1031842 6.1031842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1031842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1137984) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1137984) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1137984) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1137984) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -83.903049      0             -83.903049      593810.51    
Loop time of 7.91e-07 on 1 procs for 0 steps with 16 atoms

126.4% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1137984 6.1137984 6.1137984) with tilt (0 0 0)
            change_box all x final 0.0 6.124412633333334 y final 0.0 6.124412633333334 z final 0.0 6.124412633333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1244126 6.1137984 6.1137984) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1137984) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1244126) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1244126) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1244126) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1244126) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -84.334488      0             -84.334488      565751.86    
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1244126 6.1244126 6.1244126) with tilt (0 0 0)
            change_box all x final 0.0 6.135026866666667 y final 0.0 6.135026866666667 z final 0.0 6.135026866666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1350269 6.1244126 6.1244126) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1244126) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1350269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1350269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1350269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1350269) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -84.746558      0             -84.746558      537922.35    
Loop time of 5.84e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1350269 6.1350269 6.1350269) with tilt (0 0 0)
            change_box all x final 0.0 6.1456411 y final 0.0 6.1456411 z final 0.0 6.1456411 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1456411 6.1350269 6.1350269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1350269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1456411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1456411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1456411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1456411) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.139289      0             -85.139289      510329.99    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

160.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1456411 6.1456411 6.1456411) with tilt (0 0 0)
            change_box all x final 0.0 6.156255333333334 y final 0.0 6.156255333333334 z final 0.0 6.156255333333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1562553 6.1456411 6.1456411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1456411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1562553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1562553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1562553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1562553) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.512724      0             -85.512724      482982.77    
Loop time of 5.84e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1562553 6.1562553 6.1562553) with tilt (0 0 0)
            change_box all x final 0.0 6.1668695666666675 y final 0.0 6.1668695666666675 z final 0.0 6.1668695666666675 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1668696 6.1562553 6.1562553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1562553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1668696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1668696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1668696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1668696) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.86691       0             -85.86691       455888.63    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

184.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1668696 6.1668696 6.1668696) with tilt (0 0 0)
            change_box all x final 0.0 6.1774838 y final 0.0 6.1774838 z final 0.0 6.1774838 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1774838 6.1668696 6.1668696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1668696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1774838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1774838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1774838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1774838) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.2019        0             -86.2019        429055.51    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

149.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1774838 6.1774838 6.1774838) with tilt (0 0 0)
            change_box all x final 0.0 6.1880980333333335 y final 0.0 6.1880980333333335 z final 0.0 6.1880980333333335 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.188098 6.1774838 6.1774838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.188098 6.188098 6.1774838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.188098 6.188098 6.188098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.188098 6.188098 6.188098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.188098 6.188098 6.188098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.188098 6.188098 6.188098) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.51776       0             -86.51776       402491.36    
Loop time of 7.91e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.188098 6.188098 6.188098) with tilt (0 0 0)
            change_box all x final 0.0 6.198712266666667 y final 0.0 6.198712266666667 z final 0.0 6.198712266666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.1987123 6.188098 6.188098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.188098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.1987123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.1987123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.1987123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.1987123) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.814558      0             -86.814558      376204.15    
Loop time of 6.66e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.1987123 6.1987123 6.1987123) with tilt (0 0 0)
            change_box all x final 0.0 6.2093265 y final 0.0 6.2093265 z final 0.0 6.2093265 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2093265 6.1987123 6.1987123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.1987123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.2093265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.2093265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.2093265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.2093265) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.092375      0             -87.092375      350201.87    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

160.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2093265 6.2093265 6.2093265) with tilt (0 0 0)
            change_box all x final 0.0 6.219940733333334 y final 0.0 6.219940733333334 z final 0.0 6.219940733333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2199407 6.2093265 6.2093265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2093265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2199407) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2199407) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2199407) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2199407) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.351295      0             -87.351295      324492.6     
Loop time of 7.08e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2199407 6.2199407 6.2199407) with tilt (0 0 0)
            change_box all x final 0.0 6.230554966666666 y final 0.0 6.230554966666666 z final 0.0 6.230554966666666 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.230555 6.2199407 6.2199407) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.230555 6.230555 6.2199407) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.230555 6.230555 6.230555) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.230555 6.230555 6.230555) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.230555 6.230555 6.230555) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.230555 6.230555 6.230555) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.591415      0             -87.591415      299084.43    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

184.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.230555 6.230555 6.230555) with tilt (0 0 0)
            change_box all x final 0.0 6.2411692 y final 0.0 6.2411692 z final 0.0 6.2411692 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2411692 6.230555 6.230555) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.230555) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.2411692) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.2411692) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.2411692) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.2411692) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.812836      0             -87.812836      273985.54    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2411692 6.2411692 6.2411692) with tilt (0 0 0)
            change_box all x final 0.0 6.251783433333334 y final 0.0 6.251783433333334 z final 0.0 6.251783433333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2517834 6.2411692 6.2411692) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2411692) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2517834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2517834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2517834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2517834) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.015671      0             -88.015671      249204.17    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

171.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1792 ave        1792 max        1792 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1792
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2517834 6.2517834 6.2517834) with tilt (0 0 0)
            change_box all x final 0.0 6.262397666666667 y final 0.0 6.262397666666667 z final 0.0 6.262397666666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2623977 6.2517834 6.2517834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2517834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2623977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2623977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2623977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2623977) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.20004       0             -88.20004       224748.63    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

171.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2623977 6.2623977 6.2623977) with tilt (0 0 0)
            change_box all x final 0.0 6.2730119 y final 0.0 6.2730119 z final 0.0 6.2730119 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2730119 6.2623977 6.2623977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2623977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2730119) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2730119) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2730119) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2730119) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.36607       0             -88.36607       200627.29    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2730119 6.2730119 6.2730119) with tilt (0 0 0)
            change_box all x final 0.0 6.283626133333334 y final 0.0 6.283626133333334 z final 0.0 6.283626133333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2836261 6.2730119 6.2730119) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2730119) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2836261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2836261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2836261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2836261) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.513901      0             -88.513901      176848.59    
Loop time of 5.41e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2836261 6.2836261 6.2836261) with tilt (0 0 0)
            change_box all x final 0.0 6.294240366666667 y final 0.0 6.294240366666667 z final 0.0 6.294240366666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.2942404 6.2836261 6.2836261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2836261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2942404) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2942404) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2942404) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2942404) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.643677      0             -88.643677      153421.04    
Loop time of 5.41e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.2942404 6.2942404 6.2942404) with tilt (0 0 0)
            change_box all x final 0.0 6.3048546000000005 y final 0.0 6.3048546000000005 z final 0.0 6.3048546000000005 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3048546 6.2942404 6.2942404) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.2942404) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.3048546) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.3048546) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.3048546) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.3048546) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.755556      0             -88.755556      130353.18    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

171.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3048546 6.3048546 6.3048546) with tilt (0 0 0)
            change_box all x final 0.0 6.315468833333334 y final 0.0 6.315468833333334 z final 0.0 6.315468833333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3154688 6.3048546 6.3048546) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3048546) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3154688) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3154688) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3154688) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3154688) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.849703      0             -88.849703      107653.61    
Loop time of 5e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5e-07      |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3154688 6.3154688 6.3154688) with tilt (0 0 0)
            change_box all x final 0.0 6.3260830666666665 y final 0.0 6.3260830666666665 z final 0.0 6.3260830666666665 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3260831 6.3154688 6.3154688) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3154688) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3260831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3260831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3260831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3260831) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.926291      0             -88.926291      85330.944    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

160.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3260831 6.3260831 6.3260831) with tilt (0 0 0)
            change_box all x final 0.0 6.3366973 y final 0.0 6.3366973 z final 0.0 6.3366973 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3366973 6.3260831 6.3260831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3260831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3366973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3366973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3366973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3366973) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.985505      0             -88.985505      63393.824    
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

133.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3366973 6.3366973 6.3366973) with tilt (0 0 0)
            change_box all x final 0.0 6.347311533333333 y final 0.0 6.347311533333333 z final 0.0 6.347311533333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3473115 6.3366973 6.3366973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3366973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3473115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3473115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3473115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3473115) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.027539      0             -89.027539      41850.883    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

160.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3473115 6.3473115 6.3473115) with tilt (0 0 0)
            change_box all x final 0.0 6.357925766666667 y final 0.0 6.357925766666667 z final 0.0 6.357925766666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3579258 6.3473115 6.3473115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3473115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3579258) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3579258) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3579258) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3579258) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.052596      0             -89.052596      20710.735    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3579258 6.3579258 6.3579258) with tilt (0 0 0)
            change_box all x final 0.0 6.36854 y final 0.0 6.36854 z final 0.0 6.36854 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.36854 6.3579258 6.3579258) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.3579258) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

133.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
            change_box all x final 0.0 6.379154233333334 y final 0.0 6.379154233333334 z final 0.0 6.379154233333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3791542 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.3791542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.3791542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.3791542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.3791542) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.052643      0             -89.052643     -20326.943    
Loop time of 8.34e-07 on 1 procs for 0 steps with 16 atoms

119.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3791542 6.3791542 6.3791542) with tilt (0 0 0)
            change_box all x final 0.0 6.389768466666667 y final 0.0 6.389768466666667 z final 0.0 6.389768466666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.3897685 6.3791542 6.3791542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3791542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3897685) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3897685) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3897685) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3897685) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.028089      0             -89.028089     -40207.535    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.3897685 6.3897685 6.3897685) with tilt (0 0 0)
            change_box all x final 0.0 6.4003827 y final 0.0 6.4003827 z final 0.0 6.4003827 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4003827 6.3897685 6.3897685) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.3897685) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.4003827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.4003827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.4003827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.4003827) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.987472      0             -88.987472     -59651.505    
Loop time of 6.66e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4003827 6.4003827 6.4003827) with tilt (0 0 0)
            change_box all x final 0.0 6.410996933333333 y final 0.0 6.410996933333333 z final 0.0 6.410996933333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4109969 6.4003827 6.4003827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4003827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4109969) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4109969) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4109969) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4109969) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.931043      0             -88.931043     -78650.675    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4109969 6.4109969 6.4109969) with tilt (0 0 0)
            change_box all x final 0.0 6.421611166666667 y final 0.0 6.421611166666667 z final 0.0 6.421611166666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4216112 6.4109969 6.4109969) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4109969) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4216112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4216112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4216112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4216112) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.859067      0             -88.859067     -97197.029    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

149.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4216112 6.4216112 6.4216112) with tilt (0 0 0)
            change_box all x final 0.0 6.4322254 y final 0.0 6.4322254 z final 0.0 6.4322254 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4322254 6.4216112 6.4216112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4216112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4322254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4322254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4322254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4322254) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.771816      0             -88.771816     -115282.75    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4322254 6.4322254 6.4322254) with tilt (0 0 0)
            change_box all x final 0.0 6.4428396333333335 y final 0.0 6.4428396333333335 z final 0.0 6.4428396333333335 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4428396 6.4322254 6.4322254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4322254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4428396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4428396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4428396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4428396) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.669572      0             -88.669572     -132900.22    
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4428396 6.4428396 6.4428396) with tilt (0 0 0)
            change_box all x final 0.0 6.453453866666668 y final 0.0 6.453453866666668 z final 0.0 6.453453866666668 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4534539 6.4428396 6.4428396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4428396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4534539) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4534539) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4534539) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4534539) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.552629      0             -88.552629     -150042.11    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4534539 6.4534539 6.4534539) with tilt (0 0 0)
            change_box all x final 0.0 6.4640680999999995 y final 0.0 6.4640680999999995 z final 0.0 6.4640680999999995 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4640681 6.4534539 6.4534539) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4534539) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4640681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4640681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4640681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4640681) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.421289      0             -88.421289     -166701.34    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4640681 6.4640681 6.4640681) with tilt (0 0 0)
            change_box all x final 0.0 6.474682333333333 y final 0.0 6.474682333333333 z final 0.0 6.474682333333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4746823 6.4640681 6.4640681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4640681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4746823) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4746823) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4746823) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4746823) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.275862      0             -88.275862     -182871.15    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

171.5% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4746823 6.4746823 6.4746823) with tilt (0 0 0)
            change_box all x final 0.0 6.485296566666666 y final 0.0 6.485296566666666 z final 0.0 6.485296566666666 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4852966 6.4746823 6.4746823) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4746823) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4852966) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4852966) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4852966) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4852966) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -88.11667       0             -88.11667      -198545.11    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4852966 6.4852966 6.4852966) with tilt (0 0 0)
            change_box all x final 0.0 6.495910800000001 y final 0.0 6.495910800000001 z final 0.0 6.495910800000001 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.4959108 6.4852966 6.4852966) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4852966) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4959108) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4959108) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4959108) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4959108) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.944042      0             -87.944042     -213717.18    
Loop time of 5.42e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.4959108 6.4959108 6.4959108) with tilt (0 0 0)
            change_box all x final 0.0 6.506525033333334 y final 0.0 6.506525033333334 z final 0.0 6.506525033333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.506525 6.4959108 6.4959108) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.506525 6.506525 6.4959108) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.506525 6.506525 6.506525) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.506525 6.506525 6.506525) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.506525 6.506525 6.506525) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.506525 6.506525 6.506525) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.758317      0             -87.758317     -228381.69    
Loop time of 8.33e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.506525 6.506525 6.506525) with tilt (0 0 0)
            change_box all x final 0.0 6.517139266666668 y final 0.0 6.517139266666668 z final 0.0 6.517139266666668 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5171393 6.506525 6.506525) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.506525) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.5171393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.5171393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.5171393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.5171393) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.559842      0             -87.559842     -242533.41    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

160.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5171393 6.5171393 6.5171393) with tilt (0 0 0)
            change_box all x final 0.0 6.527753499999999 y final 0.0 6.527753499999999 z final 0.0 6.527753499999999 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5277535 6.5171393 6.5171393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5171393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5277535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5277535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5277535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5277535) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.348973      0             -87.348973     -256167.56    
Loop time of 5.41e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5277535 6.5277535 6.5277535) with tilt (0 0 0)
            change_box all x final 0.0 6.538367733333334 y final 0.0 6.538367733333334 z final 0.0 6.538367733333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5383677 6.5277535 6.5277535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5277535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5383677) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5383677) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5383677) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5383677) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -87.126073      0             -87.126073     -269279.85    
Loop time of 9.58e-07 on 1 procs for 0 steps with 16 atoms

104.4% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5383677 6.5383677 6.5383677) with tilt (0 0 0)
            change_box all x final 0.0 6.548981966666667 y final 0.0 6.548981966666667 z final 0.0 6.548981966666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.548982 6.5383677 6.5383677) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.548982 6.548982 6.5383677) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.548982 6.548982 6.548982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.548982 6.548982 6.548982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.548982 6.548982 6.548982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.548982 6.548982 6.548982) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.891513      0             -86.891513     -281866.46    
Loop time of 7.5e-07 on 1 procs for 0 steps with 16 atoms

133.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.548982 6.548982 6.548982) with tilt (0 0 0)
            change_box all x final 0.0 6.5595962000000005 y final 0.0 6.5595962000000005 z final 0.0 6.5595962000000005 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5595962 6.548982 6.548982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.548982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.5595962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.5595962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.5595962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.5595962) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.645671      0             -86.645671     -293924.15    
Loop time of 5.84e-07 on 1 procs for 0 steps with 16 atoms

171.2% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5595962 6.5595962 6.5595962) with tilt (0 0 0)
            change_box all x final 0.0 6.570210433333334 y final 0.0 6.570210433333334 z final 0.0 6.570210433333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5702104 6.5595962 6.5595962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5595962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5702104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5702104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5702104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5702104) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.388931      0             -86.388931     -305450.17    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5702104 6.5702104 6.5702104) with tilt (0 0 0)
            change_box all x final 0.0 6.580824666666667 y final 0.0 6.580824666666667 z final 0.0 6.580824666666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5808247 6.5702104 6.5702104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5702104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5808247) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5808247) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5808247) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5808247) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -86.121685      0             -86.121685     -316442.38    
Loop time of 9.59e-07 on 1 procs for 0 steps with 16 atoms

104.3% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5808247 6.5808247 6.5808247) with tilt (0 0 0)
            change_box all x final 0.0 6.5914389 y final 0.0 6.5914389 z final 0.0 6.5914389 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.5914389 6.5808247 6.5808247) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5808247) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5914389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5914389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5914389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5914389) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.84433       0             -85.84433      -326899.23    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.5914389 6.5914389 6.5914389) with tilt (0 0 0)
            change_box all x final 0.0 6.602053133333333 y final 0.0 6.602053133333333 z final 0.0 6.602053133333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6020531 6.5914389 6.5914389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.5914389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.6020531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.6020531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.6020531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.6020531) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.557268      0             -85.557268     -336819.75    
Loop time of 6.25e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6020531 6.6020531 6.6020531) with tilt (0 0 0)
            change_box all x final 0.0 6.612667366666667 y final 0.0 6.612667366666667 z final 0.0 6.612667366666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6126674 6.6020531 6.6020531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6020531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6126674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6126674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6126674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6126674) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -85.260907      0             -85.260907     -346203.62    
Loop time of 5.83e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6126674 6.6126674 6.6126674) with tilt (0 0 0)
            change_box all x final 0.0 6.6232816 y final 0.0 6.6232816 z final 0.0 6.6232816 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6232816 6.6126674 6.6126674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6126674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6232816) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6232816) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6232816) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6232816) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -84.955658      0             -84.955658     -355051.14    
Loop time of 8.34e-07 on 1 procs for 0 steps with 16 atoms

119.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6232816 6.6232816 6.6232816) with tilt (0 0 0)
            change_box all x final 0.0 6.633895833333334 y final 0.0 6.633895833333334 z final 0.0 6.633895833333334 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6338958 6.6232816 6.6232816) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6232816) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6338958) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6338958) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6338958) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6338958) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -84.641937      0             -84.641937     -363363.26    
Loop time of 5.41e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6338958 6.6338958 6.6338958) with tilt (0 0 0)
            change_box all x final 0.0 6.644510066666668 y final 0.0 6.644510066666668 z final 0.0 6.644510066666668 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6445101 6.6338958 6.6338958) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6338958) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6445101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6445101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6445101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6445101) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -84.320163      0             -84.320163     -371141.6     
Loop time of 8.34e-07 on 1 procs for 0 steps with 16 atoms

119.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6445101 6.6445101 6.6445101) with tilt (0 0 0)
            change_box all x final 0.0 6.6551243 y final 0.0 6.6551243 z final 0.0 6.6551243 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6551243 6.6445101 6.6445101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6445101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6551243) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6551243) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6551243) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6551243) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -83.990757      0             -83.990757     -378388.45    
Loop time of 7.08e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6551243 6.6551243 6.6551243) with tilt (0 0 0)
            change_box all x final 0.0 6.665738533333333 y final 0.0 6.665738533333333 z final 0.0 6.665738533333333 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6657385 6.6551243 6.6551243) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6551243) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6657385) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6657385) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6657385) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6657385) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -83.654143      0             -83.654143     -385106.75    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

149.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6657385 6.6657385 6.6657385) with tilt (0 0 0)
            change_box all x final 0.0 6.676352766666667 y final 0.0 6.676352766666667 z final 0.0 6.676352766666667 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.6763528 6.6657385 6.6657385) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6657385) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6763528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6763528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6763528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6763528) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -83.310745      0             -83.310745     -391300.15    
Loop time of 7.08e-07 on 1 procs for 0 steps with 16 atoms

0.0% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.6763528 6.6763528 6.6763528) with tilt (0 0 0)
            change_box all x final 0.0 6.686967000000001 y final 0.0 6.686967000000001 z final 0.0 6.686967000000001 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.686967 6.6763528 6.6763528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.686967 6.686967 6.6763528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.686967 6.686967 6.686967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.686967 6.686967 6.686967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.686967 6.686967 6.686967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.686967 6.686967 6.686967) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -82.96099       0             -82.96099      -396972.98    
Loop time of 6.67e-07 on 1 procs for 0 steps with 16 atoms

149.9% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1024 ave        1024 max        1024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1024
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
            

            change_box all triclinic
Changing box ...
  triclinic box = (0 0 0) to (6.686967 6.686967 6.686967) with tilt (0 0 0)
            change_box all x final 0.0 6.36854 y final 0.0 6.36854 z final 0.0 6.36854 xy final 0.0 xz final 0.0 yz final 0.0 remap units box
Changing box ...
  triclinic box = (0 0 0) to (6.36854 6.686967 6.686967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.686967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.36854 6.36854 6.36854) with tilt (0 0 0)
            run 0
WARNING: No fixes with time integration, atoms won't move (src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 77.01 | 77.01 | 77.01 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         1   0             -89.06089       0             -89.06089      -18.044       
Loop time of 5.84e-07 on 1 procs for 0 steps with 16 atoms

171.2% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             16 ave          16 max          16 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            543 ave         543 max         543 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:              0 ave           0 max           0 min
Histogram: 1 0 0 0 0 0 0 0 0 0
FullNghs:         1408 ave        1408 max        1408 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 1408
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
