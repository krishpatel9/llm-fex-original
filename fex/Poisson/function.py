import numpy as np
import torch
from torch import sin, cos, exp
import math


# THIS IS THE FUNCTION ROHAN CAME UP WITH -> THIS IS NON-SYMMETRIC
def LHS_pde(u, x, dim_set):
    v = torch.ones(u.shape).cuda()
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set).cuda()
    for i in range(dim_set):
        ux_tem = ux[:, i:i+1]
        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]
    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    return LHS

def RHS_pde(x):
    # Only simple3d case
    x0 = x[:, 0:1]
    x2 = x[:, 2:3]
    result = 16 * x2 * (4 * x2 * torch.sin(x0)**2 * torch.cos(4 * x2 * torch.cos(x0)) - torch.sin(
        4 * x2 * torch.cos(x0)) * torch.cos(x0)) + 64 * torch.cos(x0)**2 * torch.cos(4 * x2 * torch.cos(x0))
    return result

def true_solution(x):
    # Only simple3d case
    x0 = x[:, 0:1]
    x2 = x[:, 2:3]
    result = 4 * torch.cos(4 * x2 * torch.cos(x0))
    return result

# # THIS IS THE NEW 3D HELMHOLTZ FUNCTION, THAT HAS A SYMMETRIC SOLUTION
# def LHS_pde(u, x, dim_set):
#     v = torch.ones(u.shape).cuda()
#     bs = x.size(0)
    
#     # Compute the first derivatives with respect to x (spatial variables)
#     ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    
#     # Initialize a zero tensor for the second derivatives
#     uxx = torch.zeros(bs, dim_set).cuda()
    
#     # Compute second derivatives (Laplacian)
#     for i in range(dim_set):
#         ux_tem = ux[:, i:i+1]
#         uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
#         uxx[:, i] = uxx_tem[:, i]
    
#     # Laplacian term + Helmholtz term
#     laplacian = torch.sum(uxx, dim=1, keepdim=True)
#     k = 1.0  # Wave number
#     LHS = laplacian + k**2 * u
#     return LHS

# def RHS_pde(x):
#     # RHS is zero for the homogeneous Helmholtz equation
#     return torch.zeros((x.size(0), 1)).cuda()

# def true_solution(x):
#     # Helmholtz equation solution without time dependency
#     x0 = x[:, 0:1]
#     x1 = x[:, 1:2]
#     x2 = x[:, 2:3]
#     k = 1.0  # Wave number
#     result = torch.cos(k * (x0 + x1 + x2))
#     return result



# THIS IS THE ORIGINAL POISSON EQUATION BUT FOR THREE DIMENSIONS
# def LHS_pde(u, x, dim_set):
#     v = torch.ones(u.shape).cuda()  # Create a tensor of ones with the same shape as u on the GPU.
#     bs = x.size(0)  # Batch size
#     ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]  # Compute the first derivative of u with respect to x.
#     uxx = torch.zeros(bs, dim_set).cuda()  # Initialize a zero tensor for the second derivatives.
#     for i in range(dim_set):
#         ux_tem = ux[:, i:i+1]  # Get the ith partial derivative.
#         uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]  # Compute the second derivative.
#         uxx[:, i] = uxx_tem[:, i]  # Store the second derivative.
#     LHS = -torch.sum(uxx, dim=1, keepdim=True)  # Sum the second derivatives to form the LHS (Laplacian).
#     return LHS

# def RHS_pde(x):
#     bs = x.size(0)
#     return -3 * torch.ones(bs, 1).cuda()  # Poisson equation RHS for 3D, constant -3

# def true_solution(x):
#     return 0.5 * torch.sum(x**2, dim=1, keepdim=True)  # True solution: 0.5 * (x0^2 + x1^2 + x2^2)


# -----------------------------------------------------OPERATOR SECTION------------------------------------------------------------

use_original_operators = False

if use_original_operators:
    # ORIGINAL OPERATORS
    unary_functions = [
        lambda x: 0*x**2,
        lambda x: 1+0*x**2,
        lambda x: x+0*x**2,
        lambda x: x**2,
        lambda x: x**3,
        lambda x: x**4,
        torch.exp,
        torch.sin,
        torch.cos,
    ]

    binary_functions = [
        lambda x,y: x+y,
        lambda x,y: x*y,
        lambda x,y: x-y,
    ]


    unary_functions_str = [
        '({}*(0)+{})',
        '({}*(1)+{})',
        # '5',
        '({}*{}+{})',
        # '-{}',
        '({}*({})**2+{})',
        '({}*({})**3+{})',
        '({}*({})**4+{})',
        # # '({})**5',
        '({}*exp({})+{})',
        '({}*sin({})+{})',
        '({}*cos({})+{})',
        # 'ref({})',
        # 'exp(-({})**2/2)']
    ]

    unary_functions_str_leaf= [
        '(0)',
        '(1)',
        # '5',
        '({})',
        # '-{}',
        '(({})**2)',
        '(({})**3)',
        '(({})**4)',
        # '({})**5',
        '(exp({}))',
        '(sin({}))',
        '(cos({}))',
    ]


    binary_functions_str = [
        '(({})+({}))',
        '(({})*({}))',
        '(({})-({}))',
    ]
else : 
    # MODIFIED OPERATORS
    unary_functions = [
        # lambda x: 0*x**2,
        # lambda x: 1+0*x**2,
        lambda x: x+0*x**2,
        # lambda x: x**2,
        # lambda x: x**3,
        # lambda x: x**4,
        # torch.exp,
        # torch.sin,
        torch.cos,
    ]

    binary_functions = [
        # lambda x,y: x+y,
        lambda x,y: x*y,
        # lambda x,y: x-y,
    ]


    unary_functions_str = [
        # '({}*(0)+{})',
        # '({}*(1)+{})',
        # '5',
        '({}*{}+{})',
        # '-{}',
        # '({}*({})**2+{})',
        # '({}*({})**3+{})',
        # '({}*({})**4+{})',
        # # '({})**5',
        # '({}*exp({})+{})',
        # '({}*sin({})+{})',
        '({}*cos({})+{})',
        # 'ref({})',
        # 'exp(-({})**2/2)']
    ]

    unary_functions_str_leaf= [
        # '(0)',
        # '(1)',
        # '5',
        '({})',
        # '-{}',
        # '(({})**2)',
        # '(({})**3)',
        # '(({})**4)',
        # '({})**5',
        # '(exp({}))',
        # '(sin({}))',
        '(cos({}))',
    ]


    binary_functions_str = [
        # '(({})+({}))',
        '(({})*({}))',
        # '(({})-({}))',
    ]


if __name__ == '__main__':
    batch_size = 200
    left = -1
    right = 1
    points = (torch.rand(batch_size, 1)) * (right - left) + left
    x = torch.autograd.Variable(points.cuda(), requires_grad=True)
    function = true_solution

    '''
    PDE loss
    '''
    LHS = LHS_pde(function(x), x)
    RHS = RHS_pde(x)
    pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    '''
    boundary loss
    '''
    bc_points = torch.FloatTensor([[left], [right]]).cuda()
    bc_value = true_solution(bc_points)
    bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    print('pde loss: {} -- boundary loss: {}'.format(pde_loss.item(), bd_loss.item()))