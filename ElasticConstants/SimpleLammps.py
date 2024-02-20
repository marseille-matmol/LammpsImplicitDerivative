from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lgmres,lsqr,bicg,bicgstab
import numpy as np
from lammps import lammps
from tqdm import tqdm

			
class SimpleLammps:
	def __init__(self,
			potential,
			ftol=1.0e-3,
			start_script="""
			region C block 0 2 0 2 0 2 units lattice
			create_box 2 C
			create_atoms 1 region C
			set group all type/fraction 2 0.1 12393
			""",
			write_dat=False):
		
		"""Simple LAMMPS simulator
		
		Parameters
		----------
			potential : dictionary of potential data (see above)
			start_script : various commands for LAMMPS
			lattice : crystal lattice
			ftol : force minimization threshold
			write_dat : write a lammps dat file of first minimia
		"""
		
		# extract SNAP coefficients (our Theta)
		self.pot = potential.copy()
		self.Theta = self.write_snap_coeff_file()
		snap_coeff_file=self.pot['file_name']+'.snapcoeff'
		snap_param_file=self.pot['file_name']+'.snapparam'
		self.ele = self.pot['ele']
		self.a_lat = self.pot['a_lat']
		self.lattice = self.pot['lattice']

		self.ftol = ftol

		self.L = lammps(cmdargs=['-screen','none','-log','none'])
		

		self.L.commands_string(f"""
			clear
			atom_modify map array sort 0 0.0

			# Initialize simulation
			units           metal
			
			lattice {self.lattice} {self.a_lat} origin 0.001 0.001 0.001
			
			{start_script}
			
			mass * 184.
			
			pair_style snap
			pair_coeff * * {snap_coeff_file} {snap_param_file} {self.ele}
			#fix rc all recenter INIT INIT INIT

			compute D all sna/atom {self.pot['sna_params']}
			compute dD all snad/atom {self.pot['sna_params']}

			change_box all triclinic 
						
			#minimize 0 {self.ftol} 100 100
			
			{"write_data test.dat" if write_dat else ""}
			
			run 0
		
		""")
		
		# last x : 
		self.Ndesc = self.Theta.size//2
		# notation from writeup
		
		self.s = np.ctypeslib.as_array(self.L.gather("type",0,1)).flatten()
		self.x = np.ctypeslib.as_array(self.L.gather("x",1,3)).flatten()        
		self.f0 = np.ctypeslib.as_array(self.L.gather("f",1,3)).flatten()
		self.N = self.x.size
		self.step = 0
		self.dX = None
		self.cell = self.get_cell().copy()
		self.g, self.C = self.get_D_dD(return_dD=True)
	

	def write_snap_coeff_file(self,data=None):
		if data is None:
			data = self.pot
		f = open(data['file_name']+".snapparam",'w')
		f.write(data['params']) 
		f.close()
		header = """
		"""
		nele = 2
		ntheta = data['A']['theta'].size+1
		f = open(data['file_name']+".snapcoeff",'w')
		f.write(f"# SNAP coeffs for AB\n#TDS/IM 2023\n\n{nele} {ntheta}")
		for AB in ['A','B']:
			f.write(f"\n{data[AB]['weights']}")
			f.write(f"\n{data[AB]['offset']}")
			for T in data[AB]['theta']:
				f.write(f"\n{T}")
		f.close()
		T = np.append(data['A']['theta'],data['B']['theta'])
		return T

	def get_D_dD(self,return_dD=False):

		
		g = np.ctypeslib.as_array(
				self.L.gather("c_D",1,self.Ndesc)
			).reshape((-1,self.Ndesc))
			
		g = np.append(g[self.s==1].sum(0),g[self.s==2].sum(0))
		if not return_dD:
			return g
		
		#dF = np.einsum('ijkl,jl->ik',sys['pre_dD'],dw).flatten()


		C = np.ctypeslib.as_array(
				self.L.gather("c_dD",1,3*2*self.Ndesc)
		).reshape((-1,2,3,self.Ndesc))

		C_A = C[:,0,:,:].reshape((-1,self.Ndesc))
		C_B = C[:,1,:,:].reshape((-1,self.Ndesc))

		
		return g,np.append(C_A,C_B,axis=1)
		
	def forces(self,dx,alpha = 0.1):
		"""
			Evaluate forces for given position
			Uses [F(X+alpha * dX)-F(X) ] /alpha -> H.dX as alpha -> 0
		"""
		x = self.x + alpha*dx.flatten()
		self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(x))
		self.L.command("run 0") # need to speed up...
		self.step += 1
		f = np.ctypeslib.as_array(self.L.gather("f",1,3)).flatten()
		return f
	
	def implicit_derivative(self,method="LinearSolve",alpha=0.002,maxiter=1000):
		"""Evaluation of implicit position derivative
		
		Parameters
		----------
		method : str, optional
			"Hessian":
				Calculate Hessian, solve H.dx + c = 0
			"LinearOperator" :
				Iterative Linear solution for H.dx + c = 0,
				using LinearOperator L(dx) = [F(x0)-F(x0+a*dx)]/a -> H.dx 

			 "EnergyFix" : 
				Add term c.dx to LAMMPS, then minimize E(x0+dx)+c.dx, giving
				-F(x0+dx) + c = H.dx + c = 0, as required
		
		Returns
		-------
			dX: numpy array (Ndesc,3N) implicit derivative
			also stores self.steps for iterative solutions
		"""
		

		assert method in ["Hessian","LinearOperator","EnergyFix"]


		if method=="Hessian":
				# solve
				self.H = self.hessian()
				# regularize
				self.H += np.eye(self.H.shape[0])*np.diag(self.H).min()*0.01
				# store internally
				self.dX = np.linalg.solve(self.H,self.C).T
				
		elif method=="LinearOperator":   
			dX0 = np.zeros_like(self.x)
			F0 = self.forces(dX0,alpha) # reinitialize LAMMPS

			# Linear operator from matrix-vector product function
			def mv(dx):
				alpha_x = alpha/max(np.abs(dx).max(),1.0)
				return (F0-self.forces(dx,alpha_x))/alpha_x
			L = LinearOperator((self.N,self.N),matvec=mv,rmatvec=mv)
			
			# One linear solution per parameter
			self.dX = []
			for c in tqdm(self.C.T):
				dX += [bicg(L,c,x0=np.zeros_like(c),maxiter=maxiter)[0]]
		
			self.dX = np.array(self.dX)

		elif method=="EnergyFix":
			# add per atom property to store force vectors 
			# and compute displacements
			# ensure we have a dynamic minimization scheme
			self.L.commands_string("""
				fix cv all property/atom d_cx d_cy d_cz
				compute d all displace/atom
				min_style fire
			""")
			self.dX = []
			for c in tqdm(self.C.T):
				# reset to initial minima
				self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(self.x))
				# set per atom force vectors
				for i,c_v in enumerate(["d_cx","d_cy","d_cz"]):
					v = c.reshape((-1,3))[:,i].astype(np.float64)*alpha
					self.L.scatter(c_v,1,1,np.ctypeslib.as_ctypes(v))
				
				# apply force fixes and minimize
				self.L.commands_string(f"""
					compute c all property/atom d_cx d_cy d_cz
					variable fx atom c_c[1]
					variable fy atom c_c[2]
					variable fz atom c_c[3]
					variable e atom v_fx*c_d[1]+v_fy*c_d[2]+v_fz*c_d[3]
					fix f all addforce v_fx v_fy v_fz energy v_e
					run 0
					minimize 0. 1e-10 {maxiter} {maxiter}
				""")
				x = np.ctypeslib.as_array(self.L.gather("x",1,3)).flatten()
				self.dX += [(x-self.x)/alpha]
				# disable commands
				self.L.commands_string("""
					uncompute c 
					unfix f
				""")
			self.L.commands_string("""
				variable fx delete
				variable fy delete
				variable fz delete
			""")
		self.dX = np.array(self.dX)
		return self.dX

	
	def hessian(self,dx=0.001):
		"""Naive calculation of the hessian

		Parameters
		----------
		dx : float, optional
			finite difference, by default 0.01
		"""
		H = np.zeros((self.N,self.N))

		dx_v = np.zeros_like(self.x.flatten())
		# 
		for i in tqdm(range(self.N)):

			dx_v[i] = dx
			H[i] = -self.forces(dx_v,alpha=1.0)
			
			dx_v *= -1.0
			H[i] -= -self.forces(dx_v,alpha=1.0)
			
			dx_v[i] = 0.0

		H /= 2.0*dx
	
		return H
			
	def perturb(self,dTheta,return_X=True,method="Hessian",reg=0.01,redo=False):
		"""Evaluate energy and position perturbations

		Parameters
		----------
		dTheta : vector of parameter perturbations
		return_X : bool, return position perturbations or not

		Returns dE and dX
		"""
		if self.dX is None or redo:
			self.dX = self.implicit_derivative(method)
		
		dX = dTheta@self.dX
		dE = self.g@dTheta - 0.5 * np.dot(self.C@dTheta,dX)
		if return_X:
			return dE,dX
		else:
			return dE
		
	def exact_perturb(self,dTheta,potential,return_X=True):
		self.write_snap_coeff_file(potential)
		file = potential['file_name']
		self.L.commands_string(f"""
			pair_style snap
			pair_coeff * * {file}.snapcoeff {file}.snapparam {self.ele}
			minimize 0 {self.ftol} 10000 10000
		""")
		x = np.ctypeslib.as_array(self.L.gather("x",1,3)).flatten()
		
		g = np.ctypeslib.as_array(
				self.L.gather("c_D",1,self.Ndesc)
			).reshape((-1,self.Ndesc))    
		g = np.append(g[self.s==1].sum(0),g[self.s==2].sum(0))
		dE = g@(self.Theta+dTheta) - self.g@self.Theta
		dX = x - self.x


		if return_X:
			return dE,dX
		else:
			return dE
		
		file = self.pot['file_name']
		self.L.commands_string(f"""
			pair_style snap
			pair_coeff * * {file}.snapcoeff {file}.snapparam {self.ele}
			minimize 0 {self.ftol} 1000 1000
		""")
		self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(self.x))
		self.L.command("run 0")
	
	def get_E(self,dX=None,dTheta=None):
		"""
			return energy D(X+dX)@(T+dT)
		"""
		if not dX is None:
			x = self.x.copy()
			x += dX.flatten()
			self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(x))
			self.L.command("run 0")
		
		g = self.get_D_dD()
		if not dX is None:
			self.L.scatter("x",1,3,np.ctypeslib.as_ctypes(self.x))
			self.L.command("run 0")
		
		E = g@self.Theta
		if dTheta is not None:
			E += g@dTheta
		return E
	
	def get_cell(self,origin=False):
		"""
			Return cell matrix and origin (optional)
		"""
		boxlo,boxhi,xy,yz,xz,_,_ = self.L.extract_box()
		cell_matrix = np.zeros((3,3))
		cell_origin = np.zeros(3)
		for cell_j in range(3):
			cell_matrix[cell_j][cell_j] = boxhi[cell_j]-boxlo[cell_j]
			cell_origin[cell_j] = boxlo[cell_j]
		cell_matrix[0][1] = xy
		cell_matrix[0][2] = xz
		cell_matrix[1][2] = yz

		if origin:
			return cell_matrix.copy(),cell_origin.copy()
		else:
			return cell_matrix.copy()

	def apply_strain(self,epsilon=np.zeros((3,3))):
		"""
			Apply strain to the system
		"""
		C = np.dot(self.get_cell(), ( np.eye(3) + epsilon ))
		self.L.commands_string(f"""
			change_box all x final 0.0 {C[0,0]} y final 0.0 {C[1,1]} z final 0.0 {C[2,2]} xy final {C[0,1]} xz final {C[0,2]} yz final {C[1,2]} remap units box
			run 0
		""")
		
		

	def get_dilation_dD(self,epsilon=0.005):
		"""
			Return dD/dV = change in D under change in supercell volume
		"""
		self.apply_strain(epsilon*np.eye(3)) # C(1+s)= C(1+e) => s=e
		D = 0.5*self.get_D_dD(return_dD=False)
		
		# C(1+e)(1+s) = C(1-e) => s = -2e/(1+e)
		self.apply_strain(-2.0*epsilon/(1.0+epsilon)*np.eye(3))
		D -= 0.5*self.get_D_dD(return_dD=False)
		
		# C(1-e)(1+s) = C => s = e/(1-e)
		self.apply_strain(epsilon/(1.0-epsilon)*np.eye(3))
		
		V = np.linalg.det(self.get_cell())

		V *= (1.0+epsilon)**3 - 1.0

		return D/V
