"""
benchmark suite comparing numeric_integrator with SciPy

compares performance and accuracy of integration and ODE solving methods
"""

import sys
sys.path.insert(0, '..')

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as scipy_integrate
from numeric_integrator import integrate, solve_ode


def benchmark_integration():
    """benchmark integration methods against scipy.integrate.quad"""
    print("="*60)
    print("INTEGRATION BENCHMARK")
    print("="*60)
    
    # test functions with known integrals
    test_cases = [
        {
            'name': 'polynomial',
            'func': lambda x: x**3,
            'a': 0,
            'b': 2,
            'exact': 4.0
        },
        {
            'name': 'trigonometric',
            'func': np.sin,
            'a': 0,
            'b': np.pi,
            'exact': 2.0
        },
        {
            'name': 'exponential',
            'func': np.exp,
            'a': 0,
            'b': 1,
            'exact': np.e - 1
        },
        {
            'name': 'oscillatory',
            'func': lambda x: np.sin(10*x),
            'a': 0,
            'b': np.pi,
            'exact': 0.2
        }
    ]
    
    methods = ['trapezoidal', 'simpson', 'romberg', 'adaptive_simpson']
    
    for case in test_cases:
        print(f"\n{case['name'].upper()}: ∫[{case['a']},{case['b']}] f(x)dx = {case['exact']:.6f}")
        print("-" * 60)
        
        # scipy benchmark
        t0 = time.perf_counter()
        scipy_result, scipy_error = scipy_integrate.quad(case['func'], case['a'], case['b'])
        scipy_time = time.perf_counter() - t0
        scipy_actual_error = abs(scipy_result - case['exact'])
        
        print(f"{'scipy.quad':20s}: value={scipy_result:.10f}, error={scipy_actual_error:.2e}, time={scipy_time*1e6:.2f}μs")
        
        # our methods
        for method in methods:
            try:
                t0 = time.perf_counter()
                if 'adaptive' in method:
                    result = integrate(case['func'], case['a'], case['b'], method=method, tol=1e-8)
                else:
                    result = integrate(case['func'], case['a'], case['b'], method=method, n=1000)
                our_time = time.perf_counter() - t0
                
                error = abs(result.value - case['exact'])
                speedup = scipy_time / our_time if our_time > 0 else float('inf')
                
                print(f"{method:20s}: value={result.value:.10f}, error={error:.2e}, time={our_time*1e6:.2f}μs, speedup={speedup:.2f}x")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")


def benchmark_ode():
    """benchmark ODE solvers against scipy.integrate.solve_ivp"""
    print("\n" + "="*60)
    print("ODE SOLVER BENCHMARK")
    print("="*60)
    
    # test ODEs with known solutions
    test_cases = [
        {
            'name': 'exponential growth',
            'func': lambda t, y: y,
            'y0': 1.0,
            't_span': (0, 2),
            'exact': lambda t: np.exp(t)
        },
        {
            'name': 'exponential decay',
            'func': lambda t, y: -y,
            'y0': 1.0,
            't_span': (0, 2),
            'exact': lambda t: np.exp(-t)
        },
        {
            'name': 'oscillator',
            'func': lambda t, y: np.array([y[1], -y[0]]) if hasattr(y, '__len__') else -t*y,
            'y0': np.array([1.0, 0.0]),
            't_span': (0, 2*np.pi),
            'exact': lambda t: np.array([np.cos(t), -np.sin(t)])
        }
    ]
    
    methods = ['euler', 'heun', 'rk4', 'rkf45']
    scipy_methods = ['RK23', 'RK45', 'DOP853']
    
    for case in test_cases:
        if case['name'] == 'oscillator':
            # skip complex case for now
            continue
            
        print(f"\n{case['name'].upper()}: dy/dt = f(t,y), y({case['t_span'][0]})={case['y0']}")
        print("-" * 60)
        
        t0, t_end = case['t_span']
        
        # scipy benchmarks
        for scipy_method in scipy_methods:
            try:
                t_start = time.perf_counter()
                sol = scipy_integrate.solve_ivp(
                    case['func'], 
                    case['t_span'], 
                    [case['y0']] if not hasattr(case['y0'], '__len__') else case['y0'],
                    method=scipy_method,
                    rtol=1e-6,
                    atol=1e-9
                )
                scipy_time = time.perf_counter() - t_start
                
                y_exact = case['exact'](t_end)
                error = abs(sol.y[:, -1][0] - y_exact)
                
                print(f"{'scipy.' + scipy_method:20s}: y({t_end})={sol.y[:,-1][0]:.10f}, error={error:.2e}, time={scipy_time*1e6:.2f}μs, evals={sol.nfev}")
            except Exception as e:
                print(f"{'scipy.' + scipy_method:20s}: FAILED - {e}")
        
        # our methods
        for method in methods:
            try:
                t_start = time.perf_counter()
                if method == 'rkf45':
                    sol = solve_ode(case['func'], case['y0'], t0, t_end, method=method, tol=1e-6)
                else:
                    sol = solve_ode(case['func'], case['y0'], t0, t_end, method=method, step=0.01)
                our_time = time.perf_counter() - t_start
                
                y_exact = case['exact'](t_end)
                if hasattr(sol.y[-1], '__len__'):
                    y_final = sol.y[-1][0]
                else:
                    y_final = sol.y[-1]
                    
                error = abs(y_final - y_exact)
                
                print(f"{method:20s}: y({t_end})={y_final:.10f}, error={error:.2e}, time={our_time*1e6:.2f}μs, evals={sol.n_evaluations}")
            except Exception as e:
                print(f"{method:20s}: FAILED - {e}")


def convergence_study():
    """study convergence rates of different methods"""
    print("\n" + "="*60)
    print("CONVERGENCE STUDY")
    print("="*60)
    
    # integration convergence
    print("\nINTEGRATION CONVERGENCE")
    print("-" * 60)
    
    f = lambda x: np.sin(x)
    exact = 2.0
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    
    results = {
        'trapezoidal': [],
        'simpson': [],
        'midpoint': []
    }
    
    for n in n_values:
        for method in results.keys():
            try:
                result = integrate(f, 0, np.pi, method=method, n=n if method != 'simpson' else (n if n%2==0 else n+1))
                error = abs(result.value - exact)
                results[method].append(error)
            except:
                results[method].append(np.nan)
    
    # plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for method, errors in results.items():
        plt.loglog(n_values, errors, 'o-', label=method, linewidth=2, markersize=6)
    plt.loglog(n_values, [1/n**2 for n in n_values], '--', label='O(h²)', alpha=0.4)
    plt.loglog(n_values, [1/n**4 for n in n_values], '--', label='O(h⁴)', alpha=0.4)
    plt.xlabel('number of intervals (n)', fontsize=12)
    plt.ylabel('absolute error', fontsize=12)
    plt.title('integration method convergence', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # ODE convergence
    print("\nODE SOLVER CONVERGENCE")
    print("-" * 60)
    
    f_ode = lambda x, y: y
    y0 = 1.0
    x_end = 1.0
    exact_ode = np.exp(x_end)
    
    step_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    ode_results = {
        'euler': [],
        'heun': [],
        'rk4': []
    }
    
    for step in step_values:
        for method in ode_results.keys():
            try:
                sol = solve_ode(f_ode, y0, 0, x_end, method=method, step=step)
                error = abs(sol.y[-1] - exact_ode)
                ode_results[method].append(error)
            except:
                ode_results[method].append(np.nan)
    
    plt.subplot(1, 2, 2)
    for method, errors in ode_results.items():
        plt.loglog(step_values, errors, 'o-', label=method, linewidth=2, markersize=6)
    plt.loglog(step_values, step_values, '--', label='O(h)', alpha=0.4)
    plt.loglog(step_values, [h**2 for h in step_values], '--', label='O(h²)', alpha=0.4)
    plt.loglog(step_values, [h**4 for h in step_values], '--', label='O(h⁴)', alpha=0.4)
    plt.xlabel('step size (h)', fontsize=12)
    plt.ylabel('absolute error', fontsize=12)
    plt.title('ODE solver convergence', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150, bbox_inches='tight')
    print("\nconvergence plot saved as 'convergence_study.png'")
    plt.show()


def performance_comparison():
    """compare performance across problem sizes"""
    print("\n" + "="*60)
    print("PERFORMANCE SCALING")
    print("="*60)
    
    f = lambda x: np.exp(x) * np.sin(x)
    
    n_values = [100, 200, 500, 1000, 2000, 5000]
    
    times_trap = []
    times_simpson = []
    times_scipy = []
    
    print("\nintegration timing:")
    for n in n_values:
        # trapezoidal
        t0 = time.perf_counter()
        integrate(f, 0, 1, method='trapezoidal', n=n)
        times_trap.append(time.perf_counter() - t0)
        
        # simpson
        t0 = time.perf_counter()
        integrate(f, 0, 1, method='simpson', n=n if n%2==0 else n+1)
        times_simpson.append(time.perf_counter() - t0)
        
        # scipy
        t0 = time.perf_counter()
        scipy_integrate.quad(f, 0, 1)
        times_scipy.append(time.perf_counter() - t0)
        
        print(f"n={n:5d}: trap={times_trap[-1]*1e6:7.2f}μs, simpson={times_simpson[-1]*1e6:7.2f}μs, scipy={times_scipy[-1]*1e6:7.2f}μs")
    
    # plot
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, [t*1e6 for t in times_trap], 'o-', label='trapezoidal', linewidth=2, markersize=8)
    plt.loglog(n_values, [t*1e6 for t in times_simpson], 's-', label='simpson', linewidth=2, markersize=8)
    plt.loglog(n_values, [t*1e6 for t in times_scipy], '^-', label='scipy.quad', linewidth=2, markersize=8)
    plt.xlabel('problem size (n)', fontsize=12)
    plt.ylabel('execution time (μs)', fontsize=12)
    plt.title('performance scaling', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_scaling.png', dpi=150, bbox_inches='tight')
    print("\nperformance plot saved as 'performance_scaling.png'")
    plt.show()


if __name__ == '__main__':
    print("\nNUMERIC INTEGRATOR BENCHMARK SUITE")
    print("comparing against SciPy reference implementation\n")
    
    benchmark_integration()
    benchmark_ode()
    convergence_study()
    performance_comparison()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
