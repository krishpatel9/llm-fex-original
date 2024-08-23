import numpy as np
import torch
from torch import sin, cos, exp
import math
import sympy as sp
from parser import Parser

parser = Parser()

testing_fun = "simple3d"


def LHS_pde(u, x, dim_set):
    v = torch.ones(u.shape)
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set)
    for i in range(dim_set):
        ux_tem = ux[:, i:i+1]
        uxx_tem = torch.autograd.grad(
            ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]
    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    return LHS


def RHS_pde(x):
    if testing_fun == 'paper':
        bs = x.size(0)
        dim = x.size(1)
        return -dim*torch.ones(bs, 1)
    elif testing_fun == 'simple2d':
        x1 = x[:, 1:2]
        result = -16 * torch.cos(2 * x1)
        return result
    elif testing_fun == 'simple3d':
        x0 = x[:, 0:1]
        x2 = x[:, 2:3]
        result = 16 * x2 * (4 * x2 * torch.sin(x0)**2 * torch.cos(4 * x2 * torch.cos(x0)) - torch.sin(
            4 * x2 * torch.cos(x0)) * torch.cos(x0)) + 64 * torch.cos(x0)**2 * torch.cos(4 * x2 * torch.cos(x0))
        return result
    elif testing_fun == 'long3d':
        x0 = x[:, 0:1]
        x2 = x[:, 2:3]
        term1 = 128 * (x0**4 - torch.cos(x0))**2
        term2 = (12 * x0**2 + torch.cos(x0)) * (x0**4 - torch.cos(x0))
        term3 = 3 * (4 * x0**3 + torch.sin(x0))**2
        result = term1 * (term2 + term3) - 32 * torch.exp(2 * x2)
        return result
    else:
        return None


def true_solution(x):
    if testing_fun == 'paper':
        return 0.5*torch.sum(x**2, dim=1, keepdim=True)
    elif testing_fun == 'simple2d':
        x0 = x[:, 0:1]
        x1 = x[:, 1:2]
        result = -8 * torch.cos(x1)**2
        return result
    elif testing_fun == 'simple3d':
        x0 = x[:, 0:1]
        x2 = x[:, 2:3]
        result = 4 * torch.cos(4 * x2 * torch.cos(x0))
        return result
    elif testing_fun == 'long3d':
        x0 = x[:, 0:1]
        x2 = x[:, 2:3]
        result = -2 * (-2 * x0**4 + 2 * torch.cos(x0))**4 + \
            8 * torch.exp(2 * x2)
        return result
    else:
        return None


def RHS_continuous(dim):
    if testing_fun == 'paper':
        return sp.sympify(-dim)
    elif testing_fun == 'simple2d':
        sp_fun = sp.sympify("-16*(-sin(x1)^2 + cos(x1)^2)")
        print("RHS", sp_fun)
        return sp_fun
    elif testing_fun == 'simple3d':
        sp_fun = sp.sympify(
            "16*x2*(4*x2*sin(x0)**2*cos(4*x2*cos(x0)) - sin(4*x2*cos(x0))*cos(x0)) + 64*cos(x0)**2*cos(4*x2*cos(x0))")
        print("RHS", sp_fun)
        return sp_fun
    elif testing_fun == 'long3d':
        sp_fun = sp.sympify(
            "128*(x0**4 - cos(x0))**2*((12*x0**2 + cos(x0))*(x0**4 - cos(x0)) + 3*(4*x0**3 + sin(x0))**2) - 32*exp(2*x2)")
        print("RHS", sp_fun)
        return sp_fun
    else:
        return None


def boundary_continuous(dim, condition, left, right):
    f = sp.sympify(0)
    symbols = sp.symbols(f'x:{dim}')
    E = sp.symbols('E')
    # 1/2 sum(xi^2), as per paper
    for symbol in symbols:
        f += symbol ** 2
    f *= 0.5
    tokenized_bc = []
    if condition == 'D':
        bc = calculate_dirichlet(RHS_continuous(dim), dim, left, right)
        tokenized_bc.append("Dirichlet")
    elif condition == 'N':
        bc = calculate_neumann(RHS_continuous(dim), dim, left, right)
        tokenized_bc.append("Neumann")
    elif condition == 'C':
        bc = calculate_cauchy(RHS_continuous(dim), dim, left, right)
        tokenized_bc.append("Cauchy")

    for key, values in bc.items():
        tokenized_bc.append(key)
        tokenized_bc.extend(parser.get_postfix_from_str(str(values)))

    return tokenized_bc


def calculate_dirichlet(f, dim, left, right):
    symbols = sp.symbols(f'x:{dim}')
    E = sp.symbols('E')
    print(symbols)
    bc = {}
    # boundary defined above
    for symbol in symbols:
        for bound in [left, right]:
            subs = {sym: bound if sym == symbol else sp.Symbol(
                sym.name) for sym in symbols}
            f = f.subs(E, 1)
            bc[f'{symbol}={bound}'] = simplify_constants(f.subs(subs))
    return bc


def calculate_neumann(f, dim, left, right):
    symbols = sp.symbols(f'x:{dim}')
    E = sp.symbols('E')
    print(symbols)
    bc = {}
    for symbol in symbols:
        for bound in [left, right]:
            derivative = sp.diff(f, symbol)
            subs = {sym: bound if sym == symbol else sp.Symbol(
                sym.name) for sym in symbols}
            bc[f'{symbol}={bound}'] = simplify_constants(derivative.subs(subs))
    return bc


def calculate_cauchy(f, dim, left, right):
    symbols = sp.symbols(f'x:{dim}')
    E = sp.symbols('E')
    print(symbols)
    dirichlet_bc = calculate_dirichlet(f, dim, left, right)
    neumann_bc = calculate_neumann(f, dim, left, right)
    bc = {key: (str(dirichlet_bc[key]), str(neumann_bc[key]))
          for key in dirichlet_bc}
    return bc


def simplify_constants(expr):
    def truncate(val):
        return round(val)
    expr = expr.subs(sp.E, truncate(sp.N(sp.E)))
    expr = expr.subs(sp.pi, truncate(sp.N(sp.pi)))
    expr = expr.subs(sp.sin(1), truncate(sp.N(sp.sin(1))))
    expr = expr.subs(sp.cos(1), truncate(sp.N(sp.cos(1))))
    return expr


def inference_input(dim, function_type, condition, left, right):
    input = [function_type]
    if function_type == "Poisson":
        input.extend(parser.get_postfix_from_str(str(RHS_continuous(dim))))
        input.extend(boundary_continuous(dim, condition, left, right))
    return input

# paper function
# input: ['const', 'const', 'x0', '*', 'const', 'x1', '*', '+', '^2', '*']
# guess --> ['const', 'COS', 'x1', '^2', '*', 'const', '+']


# Guess, simple2d - ['const', 'COS', 'x1', '^2', '*', 'const', '+']
# True - -8cos(x1)**2

# Guess, simple3d - ['const', 'COS', 'const', 'x2', '*', 'COS', 'x0', '*', '*'] --> [const, cos, *, x]
# True - 4*cos(4*x2*cos(x0))

# Long3d: {"F": "128*(x0**4 - cos(x0))**2*((12*x0**2 + cos(x0))*(x0**4 - cos(x0)) + 3*(4*x0**3 + sin(x0))**2) - 32*exp(2*x2)", "true_solution": "-2*(-2*x0**4 + 2*cos(x0))**4 + 8*exp(2*x2)"}
# Guess: ['const', 'x2', '*', 'const', 'const', 'x0', '^4', '*', 'const', 'COS', 'x0', '*', '*', '+', 'const', '+']


unary_llm = [
    lambda x: 1+0*x**2,
    lambda x: x+0*x**2,
    lambda x: x**2,
    # lambda x: x**4,
    torch.cos,
    # torch.exp,
]
binary_llm = [
    lambda x, y: x+y,
    lambda x, y: x*y,
]
unary_llm_str = [
    '({}*(1)+{})',
    '({}*{}+{})',
    '({}*({})**2+{})',
    '({}*cos({})+{})',
    # '({}*exp({})+{})',
]
unary_llm_str_leaf = [
    '(1)',
    '({})',
    '(({})**2)',
    '(cos({}))',
    # '(exp({}))',
]

binary_llm_str = ['(({})+({}))',
                  '(({})*({}))']

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

binary_functions = [lambda x, y: x+y,
                    lambda x, y: x*y,
                    lambda x, y: x-y
                    ]

unary_functions_str = [
    '({}*(0)+{})',
    '({}*(1)+{})',
    '({}*{}+{})',
    '({}*({})**2+{})',
    '({}*({})**3+{})',
    '({}*({})**4+{})',
    '({}*exp({})+{})',
    '({}*sin({})+{})',
    '({}*cos({})+{})'
]

unary_functions_str_leaf = [
    '(0)',
    '(1)',
    '({})',
    '(({})**2)',
    '(({})**3)',
    '(({})**4)',
    '(exp({}))',
    '(sin({}))',
    '(cos({}))',
]


binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))'
                        ]

if __name__ == '__main__':
    # batch_size = 200
    # left = -1
    # right = 1
    # points = (torch.rand(batch_size, 1)) * (right - left) + left

    # x = torch.autograd.Variable(points, requires_grad=True)
    # function = true_solution

    '''
    PDE loss
    '''
    # LHS = LHS_pde(function(x), x)
    # RHS = RHS_pde(x)
    # pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    '''
    boundary loss
    '''
    # bc_points = torch.FloatTensor([[left], [right]])
    # bc_value = true_solution(bc_points)
    # bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    # print('pde loss: {} -- boundary loss: {}'.format(pde_loss.item(), bd_loss.item()))

    '''
    Continous
    '''
    # print('f: ', RHS_continuous(3))
    # print('g: ', boundary_continuous(3, 'D', left, right))
    # print('inference input: ', inference_input(3, 'Poisson', 'D', left, right))

    RHS_continuous(2)

    x = torch.tensor([[1.0, 0.5, 1], [2.0, 1.0, 0.5]],
                     dtype=torch.float32)  # sample input tensor
    result = true_solution(x)

    print("Input:")
    print(x)
    print("True Solution Output:")
    print(result)
