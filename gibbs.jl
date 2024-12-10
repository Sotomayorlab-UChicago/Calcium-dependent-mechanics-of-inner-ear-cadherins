# Collection of routines to analyze probabilities of Ca2+ occupancy states
# in the presence of uncertainty
using LinearAlgebra,RowEchelon,Random,DataFrames,StaticArrays,CSV
using Plots,ColorSchemes,Measures,Statistics,Distributions,StatsPlots
import StatsBase

#%% frc
"""
Right-hand side of linear system for Ca²⁺-site probabilities
"""
function frc(Ca::Real,prm::Dict{Symbol,T}) where T<:Real
	 val = SVector{4}(1.,# sum of probabilities is 1
		          Ca/(prm[:K₃]+Ca), # P(Site 3 bd)
		          Ca/(prm[:K₂]+Ca), # P(Site 2 bd)
		          Ca/(prm[:K₁]+Ca));# P(Site 1 bd)  
        return val
end

#%% linsys
"""
Linear part of affine system for Ca2+ likelies in terms of unlikelies
"""
function linsys()
	# Linear system for occupancy probabilities in lex order
	#   p000 p001 p011 p111 p010 p100 p101 p110
	M = SMatrix{4,8}([1   1    1    1    1    1    1    1; # sum of probabilities is 1
	     0   1    1    1    0    0    1    0; # P(Site 3 bd)
	     0   0    1    1    1    0    0    1; # P(Site 2 bd)
	     0   0    0    1    0    1    1    1]);# P(Site 1 bd)

	#  Likely occupancy case probabilities will be affine function of unlikelies
	#   Linear part
	P = SMatrix{4,4}(-rref(M)[:,5:end]);

	return P
end

#%% λ
"""
Output the iᵗʰ affine constraint defined by constraint region
Constraint is of form L*pᵤ + fval >= 0
"""
function λ(i::Integer,P::AbstractMatrix,f::AbstractVector,prm::Dict{Symbol,T}) where T<:Real
        if i <= 4
		L = SMatrix{1,4}(i==1 ? 1. : 0.,
				 i==2 ? 1. : 0.,
				 i==3 ? 1. : 0.,
				 i==4 ? 1. : 0.); fval = 0.;
	elseif i<=8
		vtemp = @view P[i-4,:];
		L = SMatrix{1,4}(vtemp); fval = f[i-4];
    	else
		L = SMatrix{1,4}(-1.,-1.,-1.,-1.); fval = prm[:τ];
	end
			    
	return L,fval
end

#%% mbr
"""
Function to say whether the given point is a member of the constraint region
"""
function mbr(p0::AbstractVector,
             P::AbstractMatrix,f::AbstractVector,prm::Dict{Symbol,T}) where T<:Real
	flagmbr = true;
	for i=1:9
		L,fval = λ(i,P,f,prm);
		flagmbr = ( (L*p0)[1] + fval ) >= 0 ? true : false;
		     
		if !flagmbr
			return false
		end
	end
										    
	return true
end

#%% linprg
"""
Function to determine the extrema of likely probabilties by a linear
programming principal
"""
function linprg(prm::Dict{Symbol,T},Carg::AbstractVector{T}) where T<:Real
	nCa = length(Carg);
	pmin = fill(Inf,(4,nCa));
	pmax = fill(-Inf,(4,nCa));
	ram = Matrix{Float64}(undef,4,4); fram = Vector{Float64}(undef,4);
	fvals = Matrix{Float64}(undef,4,nCa);

	P = linsys();

	for n=1:nCa
		# Grab forcing term for this Calcium value
		f = frc(Carg[n],prm);
		#  Adjust f for rref 
		f = SVector{4}(f[1]-f[2],
		               f[2]-f[3],
		               f[3]-f[4],
		               f[4]);
		fvals[:,n] = f[:];
		gen=[1,1,1,0];
		while !( gen[1]==9 && gen[4]==9 )
			if gen[4]!=9
				gen[4]+=1
			elseif gen[3]!=9
				gen[3]+=1; gen[4]=1;
			elseif gen[2]!=9
				gen[2]+=1; gen[3]=1; gen[4]=1;
			else
				gen[1]+=1; gen[2]=1; gen[3]=1; gen[4]=1;
			end
			i=gen[1];j=gen[2];k=gen[3];ℓ=gen[4];
			
			if !(i<j && j<k && k<ℓ)
				continue
			end
			# Build the candidate vertex matrix
			ram[1,:],fram[1] = λ(i,P,f,prm);
			ram[2,:],fram[2] = λ(j,P,f,prm);
			ram[3,:],fram[3] = λ(k,P,f,prm);
			ram[4,:],fram[4] = λ(ℓ,P,f,prm);                                                                                        
			if rank(ram)<4
				continue
			end
			nd = -ram\fram;
			
			if !mbr(nd,P,f,prm)
				continue
			end

		
            # Found a vertex so now evaluate for extrema of p
            ram[1,:],fram[1] = λ(5,P,f,prm);
            ram[2,:],fram[2] = λ(6,P,f,prm);
            ram[3,:],fram[3] = λ(7,P,f,prm);
            ram[4,:],fram[4] = λ(8,P,f,prm);

            ram[:,1] = ram*nd+fram;
		
            # Update pmin,pmax as needed
            pmin[1,n] = ram[1,1] < pmin[1,n] ? ram[1,1] : pmin[1,n];
            pmin[2,n] = ram[2,1] < pmin[2,n] ? ram[2,1] : pmin[2,n];
            pmin[3,n] = ram[3,1] < pmin[3,n] ? ram[3,1] : pmin[3,n];
            pmin[4,n] = ram[4,1] < pmin[4,n] ? ram[4,1] : pmin[4,n];

            pmax[1,n] = ram[1,1] > pmax[1,n] ? ram[1,1] : pmax[1,n];
            pmax[2,n] = ram[2,1] > pmax[2,n] ? ram[2,1] : pmax[2,n];
            pmax[3,n] = ram[3,1] > pmax[3,n] ? ram[3,1] : pmax[3,n];
            pmax[4,n] = ram[4,1] > pmax[4,n] ? ram[4,1] : pmax[4,n];
        end

	end

	return pmin,pmax
end

#%% gibbsmp!
"""
Routine to Gibbs sample the constraint region at fixed Ca²⁺ concentration
The unlikely p's are a uniform distribution conditioned on a convex K. Thus
f(pᵢ|p̂ᵢ,...) is uniform, ie a restriction onto the constraint set of the
Unif(τ-∑notpⁱ) of values satisfying the inequality constraints. 
x = p010 p100 p101 p110

Note: P*y+f>=0 when Gibbs sampling means all but one component is fixed,
      and the condition becomes a simple inequality in the variable that
      is sampled. Ie for each i, the one that's fixed,
      P[i]*y[i] + P[:,î]*y[î] + f > 0 =>
      P[i]*y[i] >= - Pbds
      and you have an upper or lower bound depending on the sign of P[i]
"""
function gibbssmp!(x::AbstractVector,
                   P::AbstractMatrix,f::AbstractVector,prm::Dict{Symbol,T};
                   rng::MersenneTwister=MersenneTwister()) where T<:Real
	y = SVector{4}(x);
	@inbounds for i=1:4
		ram = SVector{4}(i==1 ? 0. : 1.,
				 i==2 ? 0. : 1.,
				 i==3 ? 0. : 1.,
				 i==4 ? 0. : 1.);
		Pbds = P*(ram.*y)+f;
			b = (prm[:τ]-sum(y)+y[i]); a = 0.0;
			if P[1,i]>0
				temp = SVector{2}(a,-Pbds[1]/P[1,i]);
				a = maximum(temp);
			elseif P[1,i]<0
				temp = SVector{2}(b,-Pbds[1]/P[1,i]);
				b = minimum(temp);
			end
			if P[2,i]>0
				temp = SVector{2}(a,-Pbds[2]/P[2,i]);
				a = maximum(temp);
			elseif P[2,i]<0
				temp = SVector{2}(b,-Pbds[2]/P[2,i]);
				b = minimum(temp);
			end
			if P[3,i]>0
				temp = SVector{2}(a,-Pbds[3]/P[3,i]);
				a = maximum(temp);
			elseif P[3,i]<0
				temp = SVector{2}(b,-Pbds[3]/P[3,i]);
				b = minimum(temp);
			end
			if P[4,i]>0
				temp = SVector{2}(a,-Pbds[4]/P[4,i]);
				a = maximum(temp);
			elseif P[4,i]<0
				temp = SVector{2}(b,-Pbds[4]/P[4,i]);
				b = minimum(temp);
			end

			# Sample y
			val = a+(b-a)*rand(rng);
			y = SVector{4}(i==1 ? val : y[1],
				       i==2 ? val : y[2],
				       i==3 ? val : y[3],
				       i==4 ? val : y[4])
	end
	return y
end

#%% gibbsinit
"""
Initialize by a sample drawn by absolutely continuous distr wrt to target
"""
function gibbsinit(P::AbstractMatrix,f::AbstractVector,prm::Dict{Symbol,T};
		   rng::MersenneTwister=MersenneTwister()) where T<:Real
	y = SVector{4}(0.,0.,0.,0.);
	flagfd = false;
	# ζ var helps speed up initialization for larger values of prm[:τ]
	ζ = 0.1 <= prm[:τ] ? 0.1 : prm[:τ];
	while !flagfd
		y1 = ζ*rand(rng);
		y2 = (ζ-y1)*rand(rng)
		y3 = (ζ-y1-y2)*rand(rng)
		y4 = (ζ-y1-y2-y3)*rand(rng)
		
		y = SVector{4}(y1,y2,y3,y4);

		flagfd = mbr(y,P,f,prm);

	end

	return y
end

#%% gibbsrun
"""
Run Gibbs sampler at each Ca2+ concentration
"""
function gibbsrun(prm::Dict{Symbol,T},Carg::AbstractVector{T};
		  nsmp::Integer=25000,P::AbstractMatrix=linsys(),
                  rng::MersenneTwister=MersenneTwister()) where T<:Real
	nCa = length(Carg);
	SMP = Array{Float64,3}(undef,nCa,nsmp,8); 
	f=SVector{4}(0.,0.,0.,0.);
	gen=[1,0];
	for ℓ=1:nCa*nsmp
		    if gen[2]!=nsmp
			    gen[2]+=1;
            	else
	                    gen[1]+=1;gen[2]=1;
		    end
		    i=gen[1];j=gen[2];

		    if j==1
			    # Grab forcing term for this Calcium value
			    f = frc(Carg[i],prm);
			    #  Adjust f for rref 
			    f= SVector{4}(f[1]-f[2],
			                  f[2]-f[3],
			                  f[3]-f[4],
					  f[4]);
			    
		    	    # initialize Gibbs sample
			    SMPtemp = gibbsinit(P,f,prm;rng=rng);
			    SMP[i,j,1:4] = SMPtemp; SMP[i,j,5:8] = P*SMPtemp+f;
 
	            else
			    # Draw a Gibbs sample from previous position
			    S0temp = @view SMP[i,j-1,1:4]; S0 = SVector{4}(S0temp); 
			    Snew = gibbssmp!(S0,
				             P,f,prm;rng=rng);
			    SMP[i,j,1:4] = Snew;

			    # Compute dependent probabilities
			    SMP[i,j,5:8] = P*SMP[i,j,1:4]+f;

		    end
	end

	return SMP
end
function gibbsrun(prm::Dict{Symbol,T},Ca::Real;
                  nsmp::Integer=25000,P::AbstractMatrix=linsys(),
                  rng::MersenneTwister=MersenneTwister()) where T<:Real
	Carg = [Ca];
	SMP = gibbsrun(prm,Carg;nsmp=nsmp,P=P,rng=rng) |> (x->reshape(x,nsmp,8));

	return SMP
end

# Gelman-Rubin statistic
"""
Compute Gelman-Rubin convergence statistics for each Ca²⁺ configuration probability
"""
function grstat(SMPS0::Vector{A}) where A<:Array
    # Split each chain into a half-chain
    SMPS = vcat(SMPS0,SMPS0);
    nchains0 = length(SMPS0); 
    nCa,nsmp0,_ = size(SMPS0[1]); midnsmp0 = nsmp0÷2;
    @inbounds for ℓ=1:length(SMPS)
        if ℓ<=length(SMPS0)
            SMPS[ℓ] = SMPS[ℓ][:,1:midnsmp0,:];
        else
            SMPS[ℓ] = SMPS[ℓ][:,(midnsmp0+1):(2*midnsmp0),:];
        end
    end
    nchains = length(SMPS); 
    nCa,nsmp,_ = size(SMPS[1]);
    grs = Matrix{Float64}(undef,nCa,8);
    xbar = Vector{Float64}(undef,nchains); xvar = Vector{Float64}(undef,nchains);
    gen = [1,0];
    @inbounds for ℓ=1:nCa*8
        if gen[2]<8
            gen[2]+=1;
        else 
            gen[1]+=1; gen[2]=1;
        end
        
        Caid = gen[1]; pid = gen[2];
        
        # compute the means and variances of each chain
        @inbounds for m=1:nchains
            xsmp = SMPS[m][Caid,:,pid];
            xbar[m] = sum(xsmp)/nsmp;
            xvar[m] = 1/(nsmp-1)*sum((xsmp.-xbar[m]).^2);
        end
        
        # compute the overall mean
        xbbar = sum(xbar)/nchains;
        
        # compute the within chain variance
        W = sum(xvar)/nchains;
        
        # compute the between chain variance
        B = nsmp/(nchains-1)*sum((xbar.-xbbar).^2)
        
        # compute R̂.
        R̂ = √(( (nsmp-1)/nsmp*W+1/nsmp*B )/W);
        grs[Caid,pid] = R̂;
        
    end
        
    return grs
end

#%% Plots
"""
     pltavgbndl(mykey::String[;NL::Integer,NM::Integer,Ps::Dict])
Plot the distributions for occupancy in an average cdh23 monomer over an average bundle.

# Arguments
 - `mykey::String`: The dictionary key in Ps that you wish to plot
 - `Ps::Dict`: The likely state marginal probabilities at different [Ca²⁺] as output by the Gibbs routines,
 - `NM::Integer`: The number of cdh23 monomers in a bundle,
 - `NL::Integer`: The number of canonical linkers in a cdh23 monomer.
"""
function pltavgbndl(mykey::String;Ps::Dict,NL::Integer=NL,NM::Integer=NM)
    # histograms
    #  state 111
    Dμ = Binomial(NL,Ps[mykey][end]);
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    hst = histogram([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
               normalize=:probability,labels="111",title=mykey,
               linecolor=:white,linewidth=0,alpha=0.5)
    histogram!(xlabel="# linkers",ylabel="fraction of avg cdh23\n monomer over avg bundle");
    
    #  state 011
    Dμ = Binomial(NL,Ps[mykey][end-1]);
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
                normalize=:probability,labels="011",title=mykey,
                linecolor=:white,linewidth=0,alpha=0.5);
    
    #  state 001
    Dμ = Binomial(NL,Ps[mykey][end-2]);
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
    if j!=NM
            continue
    end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
                normalize=:probability,labels="001",title=mykey,
                linecolor=:white,linewidth=0,alpha=0.5);
    
    #  state 000
    Dμ = Binomial(NL,Ps[mykey][end-3]);
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
                normalize=:probability,labels="000",title=mykey,
                linecolor=:white,linewidth=0,alpha=0.5)
    return hst
end;

"""
     pltbndl(mykey::String[;NL::Integer,NM::Integer,Ps::Dict])
Plot the distributions for occupancy in an average cdh23 monomer over an average bundle.

# Arguments
 - `mykey::Integer`: The Ca²⁺ concentration evaluating at indexing into SMP
 - `SMP::AbstractArray`: The Gibbs samples of Ca²⁺ state probabilities,
 - `NM::Integer`: The number of cdh23 monomers in a bundle,
 - `NL::Integer`: The number of canonical linkers in a cdh23 monomer.
"""
function pltbndl(mykey::Integer;
                 SMP::AbstractArray,NL::Integer=NL,NM::Integer=NM)
    # histograms
    #  state 111
    psmp = 0.5;
    Dμ = Binomial(NL,psmp); 
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        if j==1
            psmp = rand(myrng,SMP[mykey,:,end])
            Dμ = Binomial(NL,psmp)
        end
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    hst = histogram([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
               normalize=:probability,labels="111",title=mykey,
               linecolor=:white,linewidth=0,alpha=0.5)
    histogram!(xlabel="# linkers",ylabel="fraction of avg cdh23\n monomer over avg bundle");
    
    #  state 011
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        if j==1
            psmp = rand(myrng,SMP[mykey,:,end-1])
            Dμ = Binomial(NL,psmp)
        end
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
                normalize=:probability,labels="011",title=mykey,
                linecolor=:white,linewidth=0,alpha=0.5);
    
    #  state 001
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        if j==1
            psmp = rand(myrng,SMP[mykey,:,end-2])
            Dμ = Binomial(NL,psmp)
        end
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
                normalize=:probability,labels="001",title=mykey,
                linecolor=:white,linewidth=0,alpha=0.5);
    #%%%%%
    #println("Ca2+: $mykey")
    #println(transpose(hcat([k for k∈0:1:NL],cumsum(Ŷ[:]))))
    #%%%%%
    
    #  state 000
    X = Vector{Float64}(undef,NM); Y = Matrix{Float64}(undef,NL+1,nsmp);
    for i=1:nsmp,j=1:NM
        if j==1
            psmp = rand(myrng,SMP[mykey,:,end-3])
            Dμ = Binomial(NL,psmp)
        end
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:,i] = [sum(X.==k) for k∈0:1:NL]/NM;
    end
    
    Ŷ = sum(Y,dims=2)/nsmp;
    histogram!([k for k∈0:1:NL],weights=Ŷ,bins=-0.5:1:25.5,size=(450,300),
	       normalize=:probability,labels="000",title="[Ca²⁺]= "*string(mykey)*" μM",
                linecolor=:white,linewidth=0,alpha=0.5)
    return hst
end;

"""
     pltavgbndlvar(mykey::String[;NL::Integer,NM::Integer,Ps::Dict])
Plot the scatters for occupancy variance in cdh23 monomer over an average bundle.

# Arguments
 - `mykey::String`: The dictionary key in Ps that you wish to plot
 - `Ps::Dict`: The likely state marginal probabilities at different [Ca²⁺] as output by the Gibbs routines,
 - `NM::Integer`: The number of cdh23 monomers in a bundle,
 - `NL::Integer`: The number of canonical linkers in a cdh23 monomer.
"""
function pltavgbndlvar(mykey::String;Ps::Dict,NL::Integer=NL,NM::Integer=NM)
    # histograms
    #  state 111
    Dμ = Binomial(NL,Ps[mykey][end]);
    X = Vector{Float64}(undef,NM); Y = Vector{Float64}(undef,NL+1); Z = Array{Float64,3}(undef,4,3,nsmp) 
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
        Z[1,1,i] = Statistics.mean(X);
        Z[1,2,i] = Statistics.std(X);
        Z[1,3,i] = StatsBase.skewness(X);
    end
    
   #  state 011
    Dμ = Binomial(NL,Ps[mykey][end-1]);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
        Z[2,1,i] = Statistics.mean(X);
        Z[2,2,i] = Statistics.std(X);
        Z[2,3,i] = StatsBase.skewness(X);
    end
    
   #  state 001
    Dμ = Binomial(NL,Ps[mykey][end-2]);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
        Z[3,1,i] = Statistics.mean(X);
        Z[3,2,i] = Statistics.std(X);
        Z[3,3,i] = StatsBase.skewness(X);
    end
    
    
   #  state 001
    Dμ = Binomial(NL,Ps[mykey][end-3]);
    for i=1:nsmp,j=1:NM
        X[j] = rand(myrng,Dμ);
        if j!=NM
            continue
        end
        Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
        Z[4,1,i] = Statistics.mean(X);
        Z[4,2,i] = Statistics.std(X);
        Z[4,3,i] = StatsBase.skewness(X);
    end
    
    sct = scatter(Z[1,1,:][:],Z[1,2,:][:],marker=:circle,markerstrokewidth=0,alpha=0.3,c=:jet,margin=5mm,
                  size=(450,300),marker_z=Z[1,3,:],xlabel="mean",ylabel="std",labels="111")
    return sct,Z
end;

"""
     bndlvar(Cakey::String[;NL::Integer,NM::Integer,Ps::Dict])
Plot the distributions for occupancy in an average cdh23 monomer over an average bundle.

# Arguments
 - `Cakey::Integer`: The Ca²⁺ concentration evaluating at indexing into SMP
 - `SMP::AbstractArray`: The Gibbs samples of Ca²⁺ state probabilities,
 - `NM::Integer`: The number of cdh23 monomers in a bundle,
 - `NL::Integer`: The number of canonical linkers in a cdh23 monomer.
"""
function bndlvar(Cakey::Integer;
                    SMP::AbstractArray,NL::Integer=NL,NM::Integer=NM,
                    nsmp::Integer=nsmp,
                    state::String="111")
    # histograms
    psmp = 0.5;
    Dμ = Binomial(NL,psmp); 
    X = Vector{Float64}(undef,NM); Y = Vector{Float64}(undef,NL+1); Z = Array{Float64,3}(undef,4,3,nsmp);
    #if state=="111"
        for i=1:nsmp,j=1:NM
            if j==1
                psmp = rand(myrng,SMP[Cakey,:,end])
                Dμ = Binomial(NL,psmp)
            end
            X[j] = rand(myrng,Dμ);
            if j!=NM
                continue
            end
            Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
            Z[1,1,i] = Statistics.mean(X);
            Z[1,2,i] = Statistics.std(X);
            Z[1,3,i] = StatsBase.skewness(X);
        end
    #elseif state=="011"
        for i=1:nsmp,j=1:NM
            if j==1
                psmp = rand(myrng,SMP[Cakey,:,end-1])
                Dμ = Binomial(NL,psmp)
            end
            X[j] = rand(myrng,Dμ);
            if j!=NM
                continue
            end
            Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
            Z[2,1,i] = Statistics.mean(X);
            Z[2,2,i] = Statistics.std(X);
            Z[2,3,i] = StatsBase.skewness(X);
        end
    #elseif state=="001"
        for i=1:nsmp,j=1:NM
            if j==1
                psmp = rand(myrng,SMP[Cakey,:,end-2])
                Dμ = Binomial(NL,psmp)
            end
            X[j] = rand(myrng,Dμ);
            if j!=NM
                continue
            end
            Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
            Z[3,1,i] = Statistics.mean(X);
            Z[3,2,i] = Statistics.std(X);
            Z[3,3,i] = StatsBase.skewness(X);
        end
    #elseif state=="000"
        for i=1:nsmp,j=1:NM
            if j==1
                psmp = rand(myrng,SMP[Cakey,:,end-3])
                Dμ = Binomial(NL,psmp)
            end
            X[j] = rand(myrng,Dμ);
            if j!=NM
                continue
            end
            Y[:] = [sum(X.==k) for k∈0:1:NL]/NM;
            Z[4,1,i] = Statistics.mean(X);
            Z[4,2,i] = Statistics.std(X);
            Z[4,3,i] = StatsBase.skewness(X);
        end
    #else
    #    error("not an admissible state")
    #end     
    
    return Z
end;

function pltbndlvar(state::String;Zs::AbstractArray,Cavals::AbstractVector,
                                  xlims::Tuple=(0,25),ylims::Tuple=(0,10),
                                  fontfamily::AbstractString="sans-serif",
				  mrkα::Real=1)
    mykey = state=="111" ? 1 : (state=="011" ? 2 : (state=="001" ? 3 : 4))
    
    sct = Vector{Any}(undef,4);
    for ℓ=1:4
        Ca = Cavals[ℓ]
        sct[ℓ]= scatter(Zs[ℓ][mykey,1,:][:],Zs[ℓ][mykey,2,:][:],marker=:circle,markerstrokewidth=0,alpha=mrkα,margin=5mm,
                  size=(450,300),marker_z=Zs[ℓ][mykey,3,:],xlabel="avg # linkers",ylabel="std # linkers",c=:jet,
                  labels=state,title="[Ca²⁺]="*string(Ca)*" μM",
                  xtickfontsize=10,ytickfontsize=10,fontsize=12,legendfontsize=10,titlefontsize=14,
                  xlims=xlims,ylims=ylims);
    end
    
    plot!(sct[1],xlabel="");
    plot!(sct[2],legend=false,xlabel="",ylabel="");
    plot!(sct[3],legend=false);
    plot!(sct[4],legend=false,ylabel="")
    
    lay=@layout [a b;c d]
    p = plot(sct[1],sct[2],sct[3],sct[4],size=(750,500),layout=lay,fontfamily=fontfamily)
    
    return p
end;
