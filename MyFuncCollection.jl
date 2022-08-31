# ==============================================
#        BASIC FUNCTIONS & HELPER
# ==============================================
#Linspace function similar to Matlab
#Output 1D array object. Use broadcast to get multidimension.
function linspace(start, stop, length)
   return collect(range(start, stop; length = length)) 
end;

#Broadcast addition
#Helpful for double broadcasting
⊕(x, y) = x .+ y 

#Function returining maximum absolute  value
function findabsmax(in_array; dims)
    return findmax(abs.(in_array), dims = dims)
end;

# ==============================================
#        INFLUNCE LINE CALCULATOR
# ==============================================

#Input:
#axle_load: 2D Array. {dim1: N_traffic, dim2: N_axle}
#axle_position: 2D Array. 2D Array {dim1: N_traffic, dim2: N_axle}
#prop: 1D Array {dim1: N_traffic}
#TO DO: Change slope type to ::Array{Float64,1} to allow multislope
function IL_mom_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, slope, span::Float64; IL_type = "")
    #Calculate IL of moment at pts dim3 due to axle dim2 of traffic dim1
    IL = max.(min.(axle_position .* (1 .- slope), (span .- axle_position) .* slope), 0) .* axle_load
    return sum(IL, dims = 2)[:,1] #the [:,1] supress the output into 1D array
end;

# ==============================================
#        IL_mom_fixed2_midspan_calculator()
# ==============================================
#Calculate the moment influence line at midspan of a doubly fixed beam
function IL_mom_fixed2_midspan_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64; IL_type = "")
    #Limit axle position to between 0 and span
    #This covers axles outside of the bridge
    #By changing the position to either span or 0, the IL becomes zero
    axle_position = max.(min.(axle_position, span), 0)
    #Calculate IL of moment at midspan due to axle dim2 of traffic dim1
    IL = min.((0.5/span) .* (axle_position.^2), (0.5/span) .* ((span .- axle_position).^2)) .* axle_load
    return sum(IL, dims = 2)[:,1] #the [:,1] supress the output into 1D array
end;

# ==============================================
#        IL_shear_midspan_calculator()
# ==============================================
#Calculate the shear IL at midspan of simply supported beam
#=
function IL_shear_midspan_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64, loc::String)
    if loc == "right"
        axle_position = span .- axle_position
    elseif loc == "left"
        1
    else
        error("loc argument must be either 'right' or 'left'")
    end;
    #Limit axle position to between 0 and span/2
    #This covers axles outside of the IL
    axle_position[(axle_position .< 0) .| (axle_position .> (span/2))] .= 0
    IL = (axle_position ./ L) .* axle_load
    return sum(IL, dims = 2)[:,1] #the [:,1] supress the output into 1D array
end;
=#
function IL_shear_midspan_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64; IL_type::String)
    IL1 = zeros(size(axle_position))
    IL2 = zeros(size(axle_position))
    
    #Axles with position at left of midspan
    ap_l = (axle_position .>= 0) .& (axle_position .<= (span/2))
    #Axles with position at right of midspan
    ap_r = (axle_position .> (span/2)) .& (axle_position .<= (span))
    
    #Left IL is positive
    IL1[ap_l] .= (axle_position[ap_l] ./ span) .* axle_load[ap_l]
    IL1[ap_r] .= -((span .- axle_position[ap_r]) ./ span) .* axle_load[ap_r]
        
    #Left IL is negative
    IL2[ap_l] .= -(axle_position[ap_l] ./ span) .* axle_load[ap_l]
    IL2[ap_r] .= ((span .- axle_position[ap_r]) ./ span) .* axle_load[ap_r]
    
    if IL_type == "min"
        return min.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
    elseif IL_type == "max"
        return max.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
    else
        error("IL_type argument must be either 'min' or 'max'")
    end
end;


# ==============================================
#        IL_mom_fixed2_support_calculator()
# ==============================================
#Calculate the moment influence line at support of a doubly fixed beam
#Ouput the max IL of either support
function IL_mom_fixed2_support_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64; IL_type = "")
    #Constants
    p_1 = 0.34624702512138844
    p_2 = 0.6801150389163775
    x_max = -0.13905688571330002 * span
    θ_1 = -1
    
    a= (-3*span*p_1^8*p_2*θ_1 + 2*span*p_1^8*θ_1 - 3*span*p_1^7*p_2*θ_1 + 2*span*p_1^7*θ_1 + 30*span*p_1^6*p_2^3*θ_1 - 3*span*p_1^6*p_2*θ_1 - 10*span*p_1^6*θ_1 - 30*span*p_1^5*p_2^4*θ_1 - 50*span*p_1^5*p_2^3*θ_1 + 45*span*p_1^5*p_2*θ_1 + 60*span*p_1^4*p_2^4*θ_1 - 45*span*p_1^4*p_2*θ_1 + 10*span*p_1^4*θ_1 + 3*span*p_1^3*p_2*θ_1 - 2*span*p_1^3*θ_1 - 60*span*p_1^2*p_2^4*θ_1 + 50*span*p_1^2*p_2^3*θ_1 + 3*span*p_1^2*p_2*θ_1 - 2*span*p_1^2*θ_1 + 30*span*p_1*p_2^4*θ_1 - 30*span*p_1*p_2^3*θ_1 + 3*span*p_1*p_2*θ_1 - 60*p_1^3*p_2^3*x_max + 54*p_1^3*p_2*x_max - 12*p_1^3*x_max + 75*p_1^2*p_2^4*x_max - 60*p_1^2*p_2^3*x_max - 6*p_1^2*p_2*x_max + 3*p_1^2*x_max + 75*p_1*p_2^4*x_max - 60*p_1*p_2^3*x_max - 6*p_1*p_2*x_max + 3*p_1*x_max - 60*p_2^4*x_max + 60*p_2^3*x_max - 6*p_2*x_max)/(6*span^4*p_1^8*p_2^2 - 6*span^4*p_1^8*p_2 + span^4*p_1^8 - 20*span^4*p_1^7*p_2^3 + 6*span^4*p_1^7*p_2^2 + 12*span^4*p_1^7*p_2 - 3*span^4*p_1^7 + 15*span^4*p_1^6*p_2^4 + 40*span^4*p_1^6*p_2^3 - 48*span^4*p_1^6*p_2^2 + 3*span^4*p_1^6 - 45*span^4*p_1^5*p_2^4 + 48*span^4*p_1^5*p_2^2 - 12*span^4*p_1^5*p_2 - span^4*p_1^5 + 45*span^4*p_1^4*p_2^4 - 40*span^4*p_1^4*p_2^3 - 6*span^4*p_1^4*p_2^2 + 6*span^4*p_1^4*p_2 - 15*span^4*p_1^3*p_2^4 + 20*span^4*p_1^3*p_2^3 - 6*span^4*p_1^3*p_2^2)
    b= (6*span*p_1^8*p_2^2*θ_1 - 3*span*p_1^8*θ_1 - 20*span*p_1^7*p_2^3*θ_1 + 6*span*p_1^7*p_2^2*θ_1 + 5*span*p_1^7*θ_1 + 15*span*p_1^6*p_2^4*θ_1 - 20*span*p_1^6*p_2^3*θ_1 + 6*span*p_1^6*p_2^2*θ_1 + 15*span*p_1^5*p_2^4*θ_1 + 100*span*p_1^5*p_2^3*θ_1 - 90*span*p_1^5*p_2^2*θ_1 - 120*span*p_1^4*p_2^4*θ_1 + 90*span*p_1^4*p_2^2*θ_1 - 5*span*p_1^4*θ_1 + 120*span*p_1^3*p_2^4*θ_1 - 100*span*p_1^3*p_2^3*θ_1 - 6*span*p_1^3*p_2^2*θ_1 + 3*span*p_1^3*θ_1 - 15*span*p_1^2*p_2^4*θ_1 + 20*span*p_1^2*p_2^3*θ_1 - 6*span*p_1^2*p_2^2*θ_1 - 15*span*p_1*p_2^4*θ_1 + 20*span*p_1*p_2^3*θ_1 - 6*span*p_1*p_2^2*θ_1 + 120*p_1^3*p_2^3*x_max - 108*p_1^3*p_2^2*x_max + 6*p_1^3*x_max - 150*p_1^2*p_2^4*x_max + 120*p_1^2*p_2^3*x_max + 12*p_1^2*p_2^2*x_max - 4*p_1^2*x_max + 30*p_1*p_2^4*x_max - 40*p_1*p_2^3*x_max + 12*p_1*p_2^2*x_max + 30*p_2^4*x_max - 40*p_2^3*x_max + 12*p_2^2*x_max)/(6*span^3*p_1^8*p_2^2 - 6*span^3*p_1^8*p_2 + span^3*p_1^8 - 20*span^3*p_1^7*p_2^3 + 6*span^3*p_1^7*p_2^2 + 12*span^3*p_1^7*p_2 - 3*span^3*p_1^7 + 15*span^3*p_1^6*p_2^4 + 40*span^3*p_1^6*p_2^3 - 48*span^3*p_1^6*p_2^2 + 3*span^3*p_1^6 - 45*span^3*p_1^5*p_2^4 + 48*span^3*p_1^5*p_2^2 - 12*span^3*p_1^5*p_2 - span^3*p_1^5 + 45*span^3*p_1^4*p_2^4 - 40*span^3*p_1^4*p_2^3 - 6*span^3*p_1^4*p_2^2 + 6*span^3*p_1^4*p_2 - 15*span^3*p_1^3*p_2^4 + 20*span^3*p_1^3*p_2^3 - 6*span^3*p_1^3*p_2^2)
    c= (-12*span*p_1^7*p_2^2*θ_1 + 9*span*p_1^7*p_2*θ_1 + 40*span*p_1^6*p_2^3*θ_1 - 12*span*p_1^6*p_2^2*θ_1 - 15*span*p_1^6*p_2*θ_1 - 30*span*p_1^5*p_2^4*θ_1 - 50*span*p_1^5*p_2^3*θ_1 + 60*span*p_1^5*p_2^2*θ_1 + 60*span*p_1^4*p_2^4*θ_1 - 50*span*p_1^4*p_2^3*θ_1 + 50*span*p_1^3*p_2^3*θ_1 - 60*span*p_1^3*p_2^2*θ_1 + 15*span*p_1^3*p_2*θ_1 - 60*span*p_1^2*p_2^4*θ_1 + 50*span*p_1^2*p_2^3*θ_1 + 12*span*p_1^2*p_2^2*θ_1 - 9*span*p_1^2*p_2*θ_1 + 30*span*p_1*p_2^4*θ_1 - 40*span*p_1*p_2^3*θ_1 + 12*span*p_1*p_2^2*θ_1 - 60*p_1^2*p_2^3*x_max + 72*p_1^2*p_2^2*x_max - 18*p_1^2*p_2*x_max + 75*p_1*p_2^4*x_max - 60*p_1*p_2^3*x_max - 18*p_1*p_2^2*x_max + 12*p_1*p_2*x_max - 45*p_2^4*x_max + 60*p_2^3*x_max - 18*p_2^2*x_max)/(6*span^2*p_1^7*p_2^2 - 6*span^2*p_1^7*p_2 + span^2*p_1^7 - 20*span^2*p_1^6*p_2^3 + 6*span^2*p_1^6*p_2^2 + 12*span^2*p_1^6*p_2 - 3*span^2*p_1^6 + 15*span^2*p_1^5*p_2^4 + 40*span^2*p_1^5*p_2^3 - 48*span^2*p_1^5*p_2^2 + 3*span^2*p_1^5 - 45*span^2*p_1^4*p_2^4 + 48*span^2*p_1^4*p_2^2 - 12*span^2*p_1^4*p_2 - span^2*p_1^4 + 45*span^2*p_1^3*p_2^4 - 40*span^2*p_1^3*p_2^3 - 6*span^2*p_1^3*p_2^2 + 6*span^2*p_1^3*p_2 - 15*span^2*p_1^2*p_2^4 + 20*span^2*p_1^2*p_2^3 - 6*span^2*p_1^2*p_2^2)
    d= θ_1
    j= (-3*span*p_1^6*p_2*θ_1 + 2*span*p_1^6*θ_1 + 12*span*p_1^5*p_2^2*θ_1 - 3*span*p_1^5*p_2*θ_1 - 4*span*p_1^5*θ_1 - 10*span*p_1^4*p_2^3*θ_1 - 24*span*p_1^4*p_2^2*θ_1 + 24*span*p_1^4*p_2*θ_1 + 30*span*p_1^3*p_2^3*θ_1 - 24*span*p_1^3*p_2*θ_1 + 4*span*p_1^3*θ_1 - 30*span*p_1^2*p_2^3*θ_1 + 24*span*p_1^2*p_2^2*θ_1 + 3*span*p_1^2*p_2*θ_1 - 2*span*p_1^2*θ_1 + 10*span*p_1*p_2^3*θ_1 - 12*span*p_1*p_2^2*θ_1 + 3*span*p_1*p_2*θ_1 - 30*p_1^2*p_2^2*x_max + 30*p_1^2*p_2*x_max - 5*p_1^2*x_max + 40*p_1*p_2^3*x_max - 30*p_1*p_2^2*x_max - 6*p_1*p_2*x_max + 3*p_1*x_max - 20*p_2^3*x_max + 24*p_2^2*x_max - 6*p_2*x_max)/(6*span^6*p_1^8*p_2^2 - 6*span^6*p_1^8*p_2 + span^6*p_1^8 - 20*span^6*p_1^7*p_2^3 + 6*span^6*p_1^7*p_2^2 + 12*span^6*p_1^7*p_2 - 3*span^6*p_1^7 + 15*span^6*p_1^6*p_2^4 + 40*span^6*p_1^6*p_2^3 - 48*span^6*p_1^6*p_2^2 + 3*span^6*p_1^6 - 45*span^6*p_1^5*p_2^4 + 48*span^6*p_1^5*p_2^2 - 12*span^6*p_1^5*p_2 - span^6*p_1^5 + 45*span^6*p_1^4*p_2^4 - 40*span^6*p_1^4*p_2^3 - 6*span^6*p_1^4*p_2^2 + 6*span^6*p_1^4*p_2 - 15*span^6*p_1^3*p_2^4 + 20*span^6*p_1^3*p_2^3 - 6*span^6*p_1^3*p_2^2)
    k= (6*span*p_1^7*p_2*θ_1 - 4*span*p_1^7*θ_1 - 18*span*p_1^6*p_2^2*θ_1 + 6*span*p_1^6*p_2*θ_1 + 5*span*p_1^6*θ_1 + 30*span*p_1^5*p_2^2*θ_1 - 30*span*p_1^5*p_2*θ_1 + 5*span*p_1^5*θ_1 + 15*span*p_1^4*p_2^4*θ_1 - 5*span*p_1^4*θ_1 - 45*span*p_1^3*p_2^4*θ_1 + 30*span*p_1^3*p_2*θ_1 - 5*span*p_1^3*θ_1 + 45*span*p_1^2*p_2^4*θ_1 - 30*span*p_1^2*p_2^2*θ_1 - 6*span*p_1^2*p_2*θ_1 + 4*span*p_1^2*θ_1 - 15*span*p_1*p_2^4*θ_1 + 18*span*p_1*p_2^2*θ_1 - 6*span*p_1*p_2*θ_1 + 36*p_1^3*p_2^2*x_max - 36*p_1^3*p_2*x_max + 6*p_1^3*x_max + 36*p_1^2*p_2^2*x_max - 36*p_1^2*p_2*x_max + 6*p_1^2*x_max - 60*p_1*p_2^4*x_max + 36*p_1*p_2^2*x_max + 12*p_1*p_2*x_max - 6*p_1*x_max + 30*p_2^4*x_max - 36*p_2^2*x_max + 12*p_2*x_max)/(6*span^5*p_1^8*p_2^2 - 6*span^5*p_1^8*p_2 + span^5*p_1^8 - 20*span^5*p_1^7*p_2^3 + 6*span^5*p_1^7*p_2^2 + 12*span^5*p_1^7*p_2 - 3*span^5*p_1^7 + 15*span^5*p_1^6*p_2^4 + 40*span^5*p_1^6*p_2^3 - 48*span^5*p_1^6*p_2^2 + 3*span^5*p_1^6 - 45*span^5*p_1^5*p_2^4 + 48*span^5*p_1^5*p_2^2 - 12*span^5*p_1^5*p_2 - span^5*p_1^5 + 45*span^5*p_1^4*p_2^4 - 40*span^5*p_1^4*p_2^3 - 6*span^5*p_1^4*p_2^2 + 6*span^5*p_1^4*p_2 - 15*span^5*p_1^3*p_2^4 + 20*span^5*p_1^3*p_2^3 - 6*span^5*p_1^3*p_2^2)

    #Run axle from left or right
    ap_l = axle_position
    ap_r = span .- axle_position
    
    #Ignore axles that are outside the bridge
    ap_l[(ap_l .< 0) .| (ap_l .> span)] .= 0
    ap_r[(ap_r .< 0) .| (ap_r .> span)] .= 0
    
    #Calculate IL of moment at support due to axle dim2 of traffic dim1
    IL1 = ((j.*ap_l.^6) .+ (k.*ap_l.^5) .+ (a.*ap_l.^4) .+ (b.*ap_l.^3) .+ (c.*ap_l.^2) .+ (d.*ap_l)).*axle_load
    IL2 = ((j.*ap_r.^6) .+ (k.*ap_r.^5) .+ (a.*ap_r.^4) .+ (b.*ap_r.^3) .+ (c.*ap_r.^2) .+ (d.*ap_r)).*axle_load
    return min.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
end;

    
# ========================================================
#        IL_mom_midsupport_continous_calculator()
# ========================================================
#Calculate the moment influence line at middle support of a continous beam
#Note IL is truly symmetrical, no need to run vehicle from either direction
function IL_mom_midsupport_continous_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64; IL_type = "")
    #Constants
    x_max = -0.09622504839224809 * span
    p = 0.5773502851012962
    θ_1 = -0.25000001237299485
    θ_2 = 0.5
    a= (-2*span*p^5*θ_1 - 2*span*p^5*θ_2 + 4*span*p^4*θ_1 + span*p^4*θ_2 + span*p^3*θ_2 - 4*span*p^2*θ_1 + 2*span*p*θ_1 + 5*p^2*x_max + 5*p*x_max - 4*x_max)/(span^4*p^6 - 3*span^4*p^5 + 3*span^4*p^4 - span^4*p^3)
    b= (span*p^6*θ_1 + span*p^6*θ_2 + span*p^5*θ_1 + span*p^5*θ_2 - 8*span*p^4*θ_1 - 2*span*p^4*θ_2 + 8*span*p^3*θ_1 - span*p^2*θ_1 - span*p*θ_1 - 10*p^2*x_max + 2*p*x_max + 2*x_max)/(span^3*p^6 - 3*span^3*p^5 + 3*span^3*p^4 - span^3*p^3)
    c= (-2*span*p^5*θ_1 - span*p^5*θ_2 + 4*span*p^4*θ_1 + span*p^4*θ_2 - 4*span*p^2*θ_1 + 2*span*p*θ_1 + 5*p*x_max - 3*x_max)/(span^2*p^5 - 3*span^2*p^4 + 3*span^2*p^3 - span^2*p^2)
    d= θ_1
    j= (span*p^4*θ_1 + span*p^4*θ_2 - 3*span*p^3*θ_1 - span*p^3*θ_2 + 3*span*p^2*θ_1 - span*p*θ_1 - 4*p*x_max + 2*x_max)/(span^5*p^6 - 3*span^5*p^5 + 3*span^5*p^4 - span^5*p^3)
    
    #Ignore axles outside the bridge
    axle_position[(axle_position .< 0) .| (axle_position .> (2*span))] .= 0
    #IL is symmetrical across two spans, those axles on the second span has the IL symmetrical to first span
    axle_position[(axle_position .> span) .& (axle_position .<= (2*span))] .= (2*span) .- axle_position[(axle_position .> span) .& (axle_position .<= (2*span))]
    
    #Calculate IL of moment at middle support due to axle dim2 of traffic dim1
    IL = ((j.*axle_position.^5) .+ (a.*axle_position.^4) .+ (b.*axle_position.^3) .+ (c.*axle_position.^2) .+ (d.*axle_position)) .* axle_load
    return sum(IL, dims = 2)[:,1]
end;

# ========================================================
#        IL_mom_midspan_continous_calculator()
# ========================================================
function IL_mom_midspan_continous_calculator(axle_load::Array{Float64,2}, axle_position::Array{Float64,2}, span::Float64; IL_type::String)
    γ = 0.40625
    θ_4 = -γ + 1
    x_max = γ*span/2
    p = 0.5846598495285041
    x_min = -0.04763059744947997 * span
    θ_1 = -0.09406677728659026
    θ_3 = 0.256359387420408
    j= (96.0*span*p^7*θ_1 + 864.0*span*p^7*θ_3 + 384.0*span*p^7*θ_4 - 720.0*span*p^6*θ_1 - 4752.0*span*p^6*θ_3 - 1728.0*span*p^6*θ_4 + 2232.0*span*p^5*θ_1 + 9720.0*span*p^5*θ_3 + 2880.0*span*p^5*θ_4 - 3660.0*span*p^4*θ_1 - 8748.0*span*p^4*θ_3 - 2112.0*span*p^4*θ_4 + 3348.0*span*p^3*θ_1 + 2916.0*span*p^3*θ_3 + 576.0*span*p^3*θ_4 - 1620.0*span*p^2*θ_1 + 324.0*span*p*θ_1 - 2048.0*p^7*x_max + 9984.0*p^6*x_max - 17664.0*p^5*x_max + 13568.0*p^4*x_max - 3840.0*p^3*x_max - 1296.0*p^2*x_min + 2160.0*p*x_min - 648.0*x_min)/(216.0*span^7*p^9 - 1620.0*span^7*p^8 + 5022.0*span^7*p^7 - 8235.0*span^7*p^6 + 7533.0*span^7*p^5 - 3645.0*span^7*p^4 + 729.0*span^7*p^3)
    k= (-192.0*span*p^8*θ_1 - 1728.0*span*p^8*θ_3 - 768.0*span*p^8*θ_4 + 960.0*span*p^7*θ_1 + 6048.0*span*p^7*θ_3 + 2112.0*span*p^7*θ_4 - 864.0*span*p^6*θ_1 - 432.0*span*p^6*θ_3 + 288.0*span*p^6*θ_4 - 3840.0*span*p^5*θ_1 - 21384.0*span*p^5*θ_3 - 5856.0*span*p^5*θ_4 + 11604.0*span*p^4*θ_1 + 29160.0*span*p^4*θ_3 + 6240.0*span*p^4*θ_4 - 13500.0*span*p^3*θ_1 - 11664.0*span*p^3*θ_3 - 2016.0*span*p^3*θ_4 + 7452.0*span*p^2*θ_1 - 1620.0*span*p*θ_1 + 4096.0*p^8*x_max - 12416.0*p^7*x_max - 1344.0*p^6*x_max + 37568.0*p^5*x_max - 41920.0*p^4*x_max + 14016.0*p^3*x_max + 1512.0*p^3*x_min + 3780.0*p^2*x_min - 9828.0*p*x_min + 3240.0*x_min)/(216.0*span^6*p^9 - 1620.0*span^6*p^8 + 5022.0*span^6*p^7 - 8235.0*span^6*p^6 + 7533.0*span^6*p^5 - 3645.0*span^6*p^4 + 729.0*span^6*p^3)
    m= (96.0*span*p^9*θ_1 + 864.0*span*p^9*θ_3 + 384.0*span*p^9*θ_4 + 240.0*span*p^8*θ_1 + 2160.0*span*p^8*θ_3 + 960.0*span*p^8*θ_4 - 4080.0*span*p^7*θ_1 - 23760.0*span*p^7*θ_3 - 7680.0*span*p^7*θ_4 + 12000.0*span*p^6*θ_1 + 44064.0*span*p^6*θ_3 + 11136.0*span*p^6*θ_4 - 12606.0*span*p^5*θ_1 - 16038.0*span*p^5*θ_3 - 2688.0*span*p^5*θ_4 - 1995.0*span*p^4*θ_1 - 22599.0*span*p^4*θ_3 - 4416.0*span*p^4*θ_4 + 15093.0*span*p^3*θ_1 + 15309.0*span*p^3*θ_3 + 2304.0*span*p^3*θ_4 - 11745.0*span*p^2*θ_1 + 2997.0*span*p*θ_1 - 2048.0*p^9*x_max - 5120.0*p^8*x_max + 46720.0*p^7*x_max - 72448.0*p^6*x_max + 18944.0*p^5*x_max + 30464.0*p^4*x_max - 16512.0*p^3*x_max - 7560.0*p^3*x_min + 1512.0*p^2*x_min + 15120.0*p*x_min - 5994.0*x_min)/(216.0*span^5*p^9 - 1620.0*span^5*p^8 + 5022.0*span^5*p^7 - 8235.0*span^5*p^6 + 7533.0*span^5*p^5 - 3645.0*span^5*p^4 + 729.0*span^5*p^3)
    a= (-480.0*span*p^9*θ_1 - 3456.0*span*p^9*θ_3 - 1344.0*span*p^9*θ_4 + 1824.0*span*p^8*θ_1 + 9936.0*span*p^8*θ_3 + 2976.0*span*p^8*θ_4 + 1440.0*span*p^7*θ_1 + 9072.0*span*p^7*θ_3 + 3168.0*span*p^7*θ_4 - 17592.0*span*p^6*θ_1 - 56376.0*span*p^6*θ_3 - 13056.0*span*p^6*θ_4 + 34230.0*span*p^5*θ_1 + 58320.0*span*p^5*θ_3 + 10560.0*span*p^5*θ_4 - 26388.0*span*p^4*θ_1 - 10935.0*span*p^4*θ_3 - 1440.0*span*p^4*θ_4 + 3240.0*span*p^3*θ_1 - 6561.0*span*p^3*θ_3 - 864.0*span*p^3*θ_4 + 6156.0*span*p^2*θ_1 - 2430.0*span*p*θ_1 + 7552.0*p^9*x_max - 18752.0*p^8*x_max - 18624.0*p^7*x_max + 86528.0*p^6*x_max - 73600.0*p^5*x_max + 10560.0*p^4*x_max + 6336.0*p^3*x_max + 13986.0*p^3*x_min - 15255.0*p^2*x_min - 7209.0*p*x_min + 4860.0*x_min)/(216.0*span^4*p^9 - 1620.0*span^4*p^8 + 5022.0*span^4*p^7 - 8235.0*span^4*p^6 + 7533.0*span^4*p^5 - 3645.0*span^4*p^4 + 729.0*span^4*p^3)
    b= (888.0*span*p^9*θ_1 + 4536.0*span*p^9*θ_3 + 1536.0*span*p^9*θ_4 - 5220.0*span*p^8*θ_1 - 21060.0*span*p^8*θ_3 - 5760.0*span*p^8*θ_4 + 10062.0*span*p^7*θ_1 + 29646.0*span*p^7*θ_3 + 6336.0*span*p^7*θ_4 - 1995.0*span*p^6*θ_1 - 2187.0*span*p^6*θ_3 + 192.0*span*p^6*θ_4 - 18909.0*span*p^5*θ_1 - 24057.0*span*p^5*θ_3 - 4032.0*span*p^5*θ_4 + 27000.0*span*p^4*θ_1 + 13122.0*span*p^4*θ_3 + 1728.0*span*p^4*θ_4 - 13770.0*span*p^3*θ_1 + 1215.0*span*p^2*θ_1 + 729.0*span*p*θ_1 - 8960.0*p^9*x_max + 36480.0*p^8*x_max - 43008.0*p^7*x_max - 256.0*p^6*x_max + 28416.0*p^5*x_max - 12672.0*p^4*x_max - 11340.0*p^3*x_min + 17334.0*p^2*x_min - 2430.0*p*x_min - 1458.0*x_min)/(216.0*span^3*p^9 - 1620.0*span^3*p^8 + 5022.0*span^3*p^7 - 8235.0*span^3*p^6 + 7533.0*span^3*p^5 - 3645.0*span^3*p^4 + 729.0*span^3*p^3)
    c= (-80.0*span*p^8*θ_1 - 216.0*span*p^8*θ_3 - 64.0*span*p^8*θ_4 + 552.0*span*p^7*θ_1 + 1188.0*span*p^7*θ_3 + 288.0*span*p^7*θ_4 - 1500.0*span*p^6*θ_1 - 2430.0*span*p^6*θ_3 - 480.0*span*p^6*θ_4 + 1934.0*span*p^5*θ_1 + 2187.0*span*p^5*θ_3 + 352.0*span*p^5*θ_4 - 960.0*span*p^4*θ_1 - 729.0*span*p^4*θ_3 - 96.0*span*p^4*θ_4 - 324.0*span*p^3*θ_1 + 540.0*span*p^2*θ_1 - 162.0*span*p*θ_1 + 384.0*p^8*x_max - 1856.0*p^7*x_max + 3264.0*p^6*x_max - 2496.0*p^5*x_max + 704.0*p^4*x_max + 378.0*p^2*x_min - 675.0*p*x_min + 243.0*x_min)/(24.0*span^2*p^8 - 180.0*span^2*p^7 + 558.0*span^2*p^6 - 915.0*span^2*p^5 + 837.0*span^2*p^4 - 405.0*span^2*p^3 + 81.0*span^2*p^2)
    d= θ_1
    
    #Run vehicle from left and right
    #Ignore axles outside the bridge
    ap_l = axle_position
    ap_r = 2 .*span .- axle_position
    ap_l[(ap_l .< 0) .| (ap_l .> (2*span))] .= 0
    ap_r[(ap_r .< 0) .| (ap_r .> (2*span))] .= 0
    
    IL1 = min.((j.*ap_l.^7) .+ (k.*ap_l.^6) .+ (m.*ap_l.^5) .+ (a.*ap_l.^4) .+ (b.*ap_l.^3) .+ (c.*ap_l.^2) .+ (d.*ap_l), (2 .*span .- ap_l) .* γ) .* axle_load
    IL2 = min.((j.*ap_r.^7) .+ (k.*ap_r.^6) .+ (m.*ap_r.^5) .+ (a.*ap_r.^4) .+ (b.*ap_r.^3) .+ (c.*ap_r.^2) .+ (d.*ap_r), (2 .*span .- ap_r) .* γ) .* axle_load

    if IL_type == "min"
        return min.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
    elseif IL_type == "max"
        return max.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
    else
        error("IL_type argument must be either 'min' or 'max'")
    end
end;


# ====================================
#        max_IL_shear_calculator()
# ====================================
#Function to calculate the maximum shear of a moving load across a simply supported beam/bridge
#The function simply checks the two possible cases of maximum shear at either supports
#
#INPUT:
#axle_load: 
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  The weight of the axle in the desired unit. Suggest N or kN.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#axle_spacing:
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  First entry in dim2 (column 1) should be all 0
#  Spacing of each axle from the previous axle.
#  NOT the distance from each axle to the first axle.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#span:
#  ::Float64
#  Span of the beam/bridge. ONLY 1 SPAN, no multispan. Simply supported.
#  
#OUTPUT:
#crit_shear:
#  ::Array{Float64,1}
#  The critical shear for each traffic instance
#  
function max_IL_shear_calculator(axle_load::Array{Float64,2}, axle_spacing::Array{Float64,2}, span::Float64)
    #Slope of IL at support
    slope = 1 ./ span
    #Axle distance relative to first axle
    axle_position = cumsum(axle_spacing, dims = 2);
    
    #IL1 and IL2 find shear IL at either right or left support
    IL1 = (span .- axle_position) .* slope .* axle_load
    IL2 = (span .- axle_position[:,end] .+ axle_position) .* slope .* axle_load
    return max.(sum(IL1, dims = 2)[:,1], sum(IL2, dims = 2)[:,1])
end;

# ====================================
#        crit_mom_load_pos()
# ====================================
# Function to calculate critical moment of a moving load across a simply supported beam/bridge
# The function does this by first movign the traffic load around the middle of the beam,
# where it is likely to be maximum (This can be changed by changing pts argument to cover the whole bridge).
# Then, it will zero in to the critical load position by moving the load position.
# It is possible for the function to be stuck on a local maxima. But the speed increase by doing the search and the precision it gives could be worth the tradeoff.
#  
#INPUT:
#axle_load: 
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  The weight of the axle in the desired unit. Suggest N or kN.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#axle_spacing:
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  First entry in dim2 (column 1) should be all 0
#  Spacing of each axle from the previous axle.
#  NOT the distance from each axle to the first axle.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#span:
#  ::Float64
#  Span of the beam/bridge. ONLY 1 SPAN, no multispan. Simply supported.
#  
#points:
#  ::Array{Float64,1}
#  Points along the beam where the moment will be calculated as part of the initial search. 
#  0 is on left support, 1 is on right support.
#  Default value is usually good enough
#  
#N_init_move:
#  ::Int32
#  Number of steps the moving load will be moved as part of the initial search. 26 is usually good enough. Higher number will be slower.
#  Minimum 2.
#  
#precision:
#  ::Float64
#  The precision of the final calculation. Suggest 0.001 (= 1 mm)
#  Will throw a warning if the steps necessary to reach the desired precision exceed a large amount (>100 by default)
#  Setting the precision to <= 0.0 will create autoprecision
#  Where the number of step for the final search is set to 10, and hence the final precision will be dictated by this.
#  By default, this is the behaviour (default value = -1.0)
#
#OUTPUT:
#crit_mom:
#  ::Array{Float64,1}
#  The critical moment for each traffic instance
#  
#crit_load_pos:
#  ::Array{Float64,1}
#  The location of the first axle relative to the left support, for each traffic instance
def_pts = linspace(0.4, 0.6, 101)
function  crit_mom_load_pos(axle_load::Array{Float64,2}, axle_spacing::Array{Float64,2}, span::Float64; points = def_pts, N_init_move = 26, precision = -1.0)
    #Args check
    if N_init_move < 2
        throw(DomainError(N_init_move, "N_init_move must be Int of at least 2"));
    end;
    
    N_traffic = size(axle_load)[1];
    N_axle = size(axle_load)[2];
    N_pts = size(points)[1];
    N_init_move = Int(N_init_move);
    
    #Axle distance relative to first axle
    axle_dist = cumsum(axle_spacing, dims = 2);
    
    ##Calculate the points along the beam relative to span
    pts = points .* span;
    #Calculate the slope of the IL for each pts
    slope = pts ./ span; #Right slope. Left slope is 1-slope
    
    ##Preallocate results array
    mom_arr = zeros(Float32, N_traffic, N_pts, N_init_move);
    
    ##Calculate move_offset for each moves
    last_axle_arr = axle_dist[:, N_axle];
    end_arr = transpose(repeat(pts, outer = [1, N_traffic]));
    start_arr = end_arr .- last_axle_arr;
    
    #Calculate move_offset by taking the extreme values out of all observed in the traffic
    start_init_move = minimum(start_arr);
    end_init_move = maximum(end_arr);
    move_offset = linspace(start_init_move, end_init_move, N_init_move);

    ##Construct initial position array
    ##Inside:{dim1: N_traffic, dim2: N_axle} Outside:{dim1: N_pts}
    init_pos = axle_dist;
    
    #Move the load across the bridge and record the moment along pts
    for i in 1:N_init_move
        axle_pos = init_pos .+ move_offset[i];
        tmp = broadcast((x) -> IL_mom_calculator(axle_load, axle_pos, x, span), slope);
        #Flatten the output from 1D array of 1D array into 2D array
        tmp = reduce(hcat, tmp);
        mom_arr[:,:,i] = tmp;
    end
    
    #Next, zero in to the actual maximum
    #Find the critical load position for each traffic
    crit_mom, crit_idx = findmax(mom_arr; dims =[2,3]);
    crit_idx = crit_idx[:,1,1]; #Eliminate the empty extra dimensions. From 3D to 1D.
    crit_load_pos_idx = getindex.(crit_idx,3); #Get the load position where maximum occurs
    
    finetune_offset = move_offset[crit_load_pos_idx]; #The starting point move offset for the finetune
    current_step_size = (move_offset[2]-move_offset[1]); #The starting step size
    
    #Calculate how many finetune steps required to reach desired precision
    #If precision is <=0, then calculate autoprecision
    #Else use the defined precision
    if precision <= 0
        num_step = 10;
        precision = current_step_size / (2^num_step);
    else
        #Calculate the number of steps required to reach the desired precision
        num_step = -log2(precision/current_step_size);
        if num_step<=0
            num_step = 0; #Precision was already achieved since the init_move steps were so fine
        else
            num_step = ceil(num_step);
        end;
    end;
    #Warn user if num_step is too large (say 100)
    if num_step > 100
        @warn("Large number of steps required to reach desired precision ($num_step steps). Consider reducing precision (increase the number)");
    end;
    
    #Preallocate results
    crit_mom = zeros(N_traffic);
    
    for i in 1:num_step
        if (i == 1) 
            current_slope = (axle_dist .+ finetune_offset)./span;
            #Turn to Array of Array for broadcasting
            current_slope = [Array{Float32}(current_slope[:,i]) for i in 1:size(current_slope)[2]];
            #Calculate moment under every axles
            crit_mom = broadcast((x) -> IL_mom_calculator(axle_load, (axle_dist .+ finetune_offset), x, span), current_slope);
            #dim1: N_traffic, dim2: N_axle (Moment under each axles)
            crit_mom = reduce(hcat, crit_mom);
            #Keep the maximum only for each traffic
            crit_mom = maximum(crit_mom, dims = 2)[:,1];
        end;
            
        #Calculate forward and backward axle position
        current_step_size = current_step_size ./ 2;
        forward_move_offset = finetune_offset .+ current_step_size;
        backward_move_offset = finetune_offset .- current_step_size;
        forward_axle_pos = axle_dist .+ forward_move_offset;
        backward_axle_pos = axle_dist .+ backward_move_offset;
        ##Check for bounds
        forward_axle_pos[forward_axle_pos.>span] .= span;
        backward_axle_pos[backward_axle_pos.<0] .= 0;

        ##IL slope
        forward_slope = forward_axle_pos ./ span;
        backward_slope = backward_axle_pos ./ span;
        ##Turn to Array of Array for broadcasting
        forward_slope = [Array{Float32}(forward_slope[:,i]) for i in 1:size(forward_slope)[2]];
        backward_slope = [Array{Float32}(backward_slope[:,i]) for i in 1:size(backward_slope)[2]];

        ##Calculate moment under every axles
        forward_mom = broadcast((x) -> IL_mom_calculator(axle_load, forward_axle_pos, x, span), forward_slope);
        backward_mom = broadcast((x) -> IL_mom_calculator(axle_load, backward_axle_pos, x, span), backward_slope);
        ##Convert back from 1D array of 1D array to 2D array
        ##dim1: N_traffic, dim2: N_axle (Moment under each axles)
        forward_mom = reduce(hcat, forward_mom);
        backward_mom = reduce(hcat, backward_mom);

        ##Keep the maximum only for each traffic
        forward_mom = maximum(forward_mom, dims = 2)[:,1];
        backward_mom = maximum(backward_mom, dims = 2)[:,1];

        #Combine the 1D array into a 2D array
        finetune_offset_array = hcat(finetune_offset, forward_move_offset, backward_move_offset);
        finetune_mom_array = hcat(crit_mom, forward_mom, backward_mom);

        #Retake the current_mom and finetune_offset
        crit_mom, crit_mom_idx = findmax(finetune_mom_array, dims = 2);
        finetune_offset = finetune_offset_array[crit_mom_idx];
    end;
    
    
    #Return the result
    #First output is the critical moment for each traffic (1D Array{dim1: N_Traffic}
    #Second output is the load position of the critical moment for each traffic (1D Array{dim1: N_Traffic}
    #Load position is the distance of the first axle (left most) to the left support
    return crit_mom,finetune_offset;
end;


# ====================================
#        generic_IL_optimiser()
# ====================================
# Optimise any generic IL function
# Work similarly to crit_mom_load_pos(), except it does not evaluate IL along the beam, only evaluates for a single point (i.e. a single IL function)
# Works by moving the vehicle along the beam, then zero in to the maximum it can find
#INPUT:
#axle_load: 
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  The weight of the axle in the desired unit. Suggest N or kN.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#axle_spacing:
#  ::Array{Float64,2} {dim1: N_traffic, dim2: N_axle}
#  First entry in dim2 (column 1) should be all 0
#  Spacing of each axle from the previous axle.
#  NOT the distance from each axle to the first axle.
#  Number of axle must be equal for all traffic instance!
#  For unequal number of axle, set the load of some of the axle to 0
#  
#span:
#  ::Float64
#  Span of the beam/bridge. ONLY 1 SPAN, no multispan
#  
#N_span:
#  ::Int
#  Number of span in the IL function, must be explicitly specified
#  
#IL_func:
#  ::Function
#  the IL function. Must have arguments:
#  IL_func(axle_load::Array{Float64,2}, axle_spacing::Array{Float64,2}, span::Float64; IL_type::String)
#
#N_init_move:
#  ::Int32
#  Number of steps the moving load will be moved as part of the initial search. 26 is usually good enough. Higher number will be slower.
#  Minimum 2.
#  
#precision:
#  ::Float64
#  The precision of the final calculation. Suggest 0.001 (= 1 mm)
#  Will throw a warning if the steps necessary to reach the desired precision exceed a large amount (>100 by default)
#  Setting the precision to <= 0.0 will create autoprecision
#  Where the number of step for the final search is set to 10, and hence the final precision will be dictated by this.
#  By default, this is the behaviour (default value = -1.0)
#
#IL_type:
#  ::String
#  either "min", "max", or "" empty String. This will specify the optimisation function and passed to the IL_func
#  if "min", the optim_func = findmin()
#  if "max", the optim_func = findmax()
#  if "", the optim_func = findabsmax()

#OUTPUT:
#crit_mom:
#  ::Array{Float64,1}
#  The critical moment for each traffic instance
#  
#crit_load_pos:
#  ::Array{Float64,1}
#  The location of the first axle relative to the left support, for each traffic instance

function generic_IL_optimiser(
        axle_load::Array{Float64,2}, 
        axle_spacing::Array{Float64,2}, 
        span::Float64, 
        N_span::Int,
        IL_func; 
        N_init_move = 26, 
        precision = -1.0, 
        IL_type = "")
    #Change optim_func according to IL_type
    if IL_type == "min"
        optim_func = findmin
    elseif IL_type == "max"
        optim_func = findmax
    elseif IL_type == ""
        optim_func = findabsmax
    else
        error("Invalid IL_type: either 'min', 'max', or empty String")
    end;
    
    #Args check
    if N_init_move < 2
        throw(DomainError(N_init_move, "N_init_move must be Int of at least 2"));
    end;
    
    #Dimension informations
    N_traffic = size(axle_load)[1];
    N_axle = size(axle_load)[2];

    #Axle distance relative to first axle
    axle_dist = cumsum(axle_spacing, dims = 2);

    #Move the traffic along the bridge
    #Start where first axle is to the left of the bridge, ends when the last axle leaves the right of the bridge
    start_arr = -axle_dist[:, N_axle]; #Pick the last axle, i.e. shifting vehicle to the start of the left support
    end_arr = (N_span * span); #Shift all the way to the end of the right support
    
    #Calculate move_offset by taking the extreme values out of all observed in the traffic
    start_init_move = minimum(start_arr);
    end_init_move = maximum(end_arr);
    move_offset = linspace(start_init_move, end_init_move, N_init_move);
    
    #Move the axles
    axle_pos = [axle_dist for i in 1:N_init_move] .⊕ move_offset # .⊕ double broadcast addition. See MyFuncCollection.jl for details
    
    #Calculate IL values
    LE_arr = broadcast((x) -> IL_func(axle_load, x, span; IL_type =  IL_type), axle_pos)
    ##Convert back from 1D array of 1D array to 2D array
    ##dim1:N_traffic, dim2: N_init_move
    LE_arr = reduce(hcat, LE_arr);

    #Next, zero in to the actual maximum
    #Find the critical load position for each traffic
    crit_LE, crit_idx = optim_func(LE_arr; dims =[2]);
    crit_load_pos_idx = getindex.(crit_idx,2); #Get the load position where maximum occurs
    finetune_offset = move_offset[crit_load_pos_idx]; #The starting point move offset for the finetune
    current_step_size = (move_offset[2]-move_offset[1]); #The starting step size
    
    #Calculate how many finetune steps required to reach desired precision
    #If precision is <=0, then calculate autoprecision
    #Else use the defined precision
    if precision <= 0
        num_step = 10;
        precision = current_step_size / (2^num_step);
    else
        #Calculate the number of steps required to reach the desired precision
        num_step = -log2(precision/current_step_size);
        if num_step<=0
            num_step = 0; #Precision was already achieved since the init_move steps were so fine
        else
            num_step = ceil(num_step);
        end;
    end;
    #Warn user if num_step is too large (say 100)
    if num_step > 100
        @warn("Large number of steps required to reach desired precision ($num_step steps). Consider reducing precision (increase the number)");
    end;
    
    for i in 1:num_step
        #Calculate forward and backward axle position
        current_step_size = current_step_size ./ 2;
        forward_move_offset = finetune_offset .+ current_step_size;
        backward_move_offset = finetune_offset .- current_step_size;
        forward_axle_pos = axle_dist .+ forward_move_offset;
        backward_axle_pos = axle_dist .+ backward_move_offset;
        
        #Calculate LE
        forward_LE = IL_func(axle_load, forward_axle_pos, span; IL_type =  IL_type)
        backward_LE = IL_func(axle_load, backward_axle_pos, span; IL_type =  IL_type)
        
        #Combine the 1D array into a 2D array
        finetune_offset_array = hcat(finetune_offset, forward_move_offset, backward_move_offset);
        finetune_LE_array = hcat(crit_LE, forward_LE, backward_LE);
        
        #Retake the current_mom and finetune_offset
        crit_LE, crit_LE_idx = optim_func(finetune_LE_array; dims = 2);
        finetune_offset = finetune_offset_array[crit_LE_idx];
    end;
    #Return the result
    #First output is the critical LE for each traffic (1D Array{dim1: N_Traffic}
    #Second output is the load position of the critical LE for each traffic (1D Array{dim1: N_Traffic}
    #Load position is the distance of the first axle (left most) to the left support
    return crit_LE, finetune_offset;
end;

# =================================
#       RANDOM DISTRIBUTIONS
# =================================
using StatsBase
using Distributions

#Multimodal Beta Random Sample
# All inputs except size np.array of size N, where N is the number of mode
# weight = importance of each mode, higher = more probability. Must sum up to 1
# alpha = alpha parameter of unimodal beta distribution, >0
# beta = beta parameter of unimodal beta distribution, >0
# size = the output size of iid sample. tuple
#Return sample of size = size

function rand_multimodal_beta(;weight, alpha, beta, rep = 1)
    N_MODE = size(weight)[1]
    indicator = sample(1:N_MODE, Weights(weight), rep, replace = true);
    
    #TO DO: Find a way to not use for loop
    #But performance isn't too bad though
    sple = zeros(rep)
    for i in 1:rep
        sple[i] = rand(Beta(alpha[indicator[i]], beta[indicator[i]]))
    end
    return sple
end;

#Sample Normal distribution with multiple mu and sd; rep time
#Each distribution with different parameter are distributed columnwise
#Each rep repetition distributed row wise

function sample_normal_rep(;mu, sd, rep = 1)
    return reduce(hcat, rand.(Normal.(mu, sd), rep));
end;

# =================================
#             METADATA
# =================================
# Data Type to handle metadata
# Initialise using initMetadata() function!
struct Metadata
    span
    n_month::Int
    n_day_in_month::Int
    aadt_m::Vector{Float64}
    aadt_sd::Vector{Float64}
    aadt_min::Vector{Float64}
    axle_spacing_m::Vector{Float64}
    axle_spacing_cv::Vector{Float64}
    axle_spacing_sd::Vector{Float64}
    axle_spacing_bound::Vector{Float64}
    n_axle::Int
    param_weight::Matrix{Float64}
    param_alpha::Matrix{Float64}
    param_beta::Matrix{Float64}
    w_max::Vector{Float64}
    w_min::Vector{Float64}
    w_scale::Vector{Float64}
    Sigma::Matrix{Float64}
    p_min::Float64
    p_max::Float64
    lower_x::Vector{Float64}
    upper_x::Vector{Float64}
end;

function initMetadata(;
    span,
    n_month::Int,
    n_day_in_month::Int,
    aadt_m::Vector{Float64},
    aadt_sd::Vector{Float64},
    aadt_min::Vector{Float64},
    axle_spacing_m::Vector{Float64},
    axle_spacing_cv::Vector{Float64},
    axle_spacing_sd::Vector{Float64},
    axle_spacing_bound::Vector{Float64},
    n_axle::Int,
    param_weight::Matrix{Float64},
    param_alpha::Matrix{Float64},
    param_beta::Matrix{Float64},
    w_max::Vector{Float64},
    w_min::Vector{Float64},
    w_scale::Vector{Float64},
    Sigma::Matrix{Float64},
    p_min::Float64,
    p_max::Float64,
    lower_x::Vector{Float64},
    upper_x::Vector{Float64},
    )
    
    #Check that matrix and vector dimensions are correct
    if n_axle != size(param_weight)[2]
        error("argument param_weight has incorrect axle dimension (dim 2)")
    end
    if n_axle != size(param_alpha)[2]
        error("argument param_alpha has incorrect axle dimension (dim 2)")
    end
    if n_axle != size(param_beta)[2]
        error("argument param_beta has incorrect axle dimension (dim 2)")
    end
    if n_axle != length(w_max)
        error("argument w_max has incorrect axle dimension (dim 2)")
    end
    if n_axle != length(w_min)
        error("argument w_min has incorrect axle dimension (dim 2)")
    end
    if n_axle != length(w_scale)
        error("argument w_scale has incorrect axle dimension (dim 2)")
    end
    #Check that the param_weight, alpha, and beta dimensions are compatible
    if size(param_alpha)[1] != size(param_weight)[1]
        error("argument param_weight and param_alpha has incompatible dimensions (dim 1)")
    end
    if size(param_beta)[1] != size(param_weight)[1]
        error("argument param_alpha has incorrect axle dimension (dim 2)")
    end
    #Check size of Sigma correlation matrix
    if size(Sigma)[1] != n_axle
        error("argument Sigma has incorrect dimension (dim 1 should be equal to n_axle)")
    end
    if size(Sigma)[2] != n_axle
        error("argument Sigma has incorrect dimension (dim 2 should be equal to n_axle)")
    end
    #Check size of lower and upper x
    if n_axle != length(lower_x)
        error("argument lower_x has incorrect axle dimension")
    end
    if n_axle != length(upper_x)
        error("argument upper_x has incorrect axle dimension")
    end
    
    return Metadata(
        span,
        n_month,
        n_day_in_month,
        aadt_m,
        aadt_sd,
        aadt_min,
        axle_spacing_m,
        axle_spacing_cv,
        axle_spacing_sd,
        axle_spacing_bound,
        n_axle,
        param_weight,
        param_alpha,
        param_beta,
        w_max,
        w_min,
        w_scale,
        Sigma,
        p_min,
        p_max,
        lower_x,
        upper_x,
    )
end;

function get_UB_vehicle(self::Metadata)
    axle = self.axle_spacing_m .- self.axle_spacing_bound
    weight = (self.upper_x .* self.w_scale) .+ self.w_min
    return axle, weight
end;