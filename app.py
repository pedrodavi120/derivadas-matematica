# -*- coding: utf-8 -*-
"""
Trabalho de Matemática Aplicada II - TADS
Assunto: Aplicações Computacionais de Derivadas
Grupo: Pedro Davi, Thiago Lima e Gustavo José.
Professora: Tásia do Vale

Este script implementa as soluções para os 5 problemas propostos
no documento "Aplicacoes_Derivadas_TI_TADS.pdf", utilizando
Python, SymPy, NumPy e Matplotlib.

Instruções para executar:
1. Certifique-se de ter as bibliotecas instaladas:
   pip install sympy numpy matplotlib
2. Execute este script. Cada problema irá gerar um gráfico.
   (Recomendado: Executar em um ambiente como Jupyter Notebook,
    copiando e colando cada "Problema X" em uma célula separada
    para melhor visualização e análise).
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Configurações globais para os gráficos (opcional, para melhor visualização)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True


def problema_1():
    """
    PROBLEMA 1: Otimização de Desempenho
    Função: T(x) = 1000/x + 2x
    Objetivo: Encontrar o ponto ótimo (mínimo)
    """
    print("--- Problema 1: Otimização de Desempenho ---")

    # 1. Análise Simbólica com SymPy
    x = sp.symbols('x')
    T = 1000/x + 2*x

    print(f"Função: T(x) = {T}")

    # Calcular a primeira derivada (T')
    T_prime = sp.diff(T, x)
    print(f"Primeira Derivada: T'(x) = {T_prime}")

    # Encontrar os pontos críticos (onde T' = 0)
    critical_points = sp.solve(T_prime, x)
    print(f"Pontos críticos (T' = 0): {critical_points}")

    # Filtrar apenas pontos reais e positivos (contexto do problema)
    # sqrt(500) = 10*sqrt(5) ≈ 22.36
    optimal_x = None
    for p in critical_points:
        if p.is_real and p > 0:
            optimal_x = p
            break
    
    if optimal_x is None:
        print("Nenhum ponto ótimo real e positivo encontrado.")
        return

    print(f"Ponto ótimo (x > 0): x = {optimal_x.evalf():.4f}")

    # Calcular a segunda derivada (T'') para verificar se é mínimo ou máximo
    T_double_prime = sp.diff(T_prime, x)
    print(f"Segunda Derivada: T''(x) = {T_double_prime}")

    # Teste da segunda derivada
    concavidade = T_double_prime.subs(x, optimal_x)
    print(f"Valor da T'' no ponto ótimo: {concavidade.evalf():.4f}")
    if concavidade > 0:
        print("Como T'' > 0, o ponto é um MÍNIMO LOCAL.")
    elif concavidade < 0:
        print("Como T'' < 0, o ponto é um MÁXIMO LOCAL.")
    else:
        print("Como T'' = 0, o teste é inconclusivo.")

    # 2. Visualização com Matplotlib e NumPy
    
    # Converter função simbólica do SymPy para uma função numérica do NumPy
    T_func = sp.lambdify(x, T, 'numpy')

    # Gerar valores de x para o gráfico
    x_vals = np.linspace(1, 50, 400) # Intervalo de x=1 a x=50
    T_vals = T_func(x_vals)
    
    plt.figure() # Criar uma nova figura
    plt.plot(x_vals, T_vals, label=f"T(x) = {sp.latex(T)}")
    
    # Marcar o ponto mínimo
    min_x = optimal_x.evalf()
    min_T = T_func(min_x)
    plt.plot(min_x, min_T, 'ro', label=f'Ponto Mínimo (x ≈ {min_x:.2f}, T ≈ {min_T:.2f})')
    
    plt.title('Problema 1: Otimização de T(x)')
    plt.xlabel('x (Recurso)')
    plt.ylabel('T(x) (Tempo/Custo)')
    plt.legend()
    plt.grid(True)
    print("Gráfico do Problema 1 gerado.")


def problema_2():
    """
    PROBLEMA 2: Otimização de Função de Custo (Gradient Descent)
    Função: J(y) = (y - 5)^2 + 3
    Objetivo: Encontrar o mínimo analiticamente e via Gradient Descent.
    """
    print("\n--- Problema 2: Otimização de Função de Custo (Gradient Descent) ---")

    # --- Fase 1: Análise Matemática (SymPy) ---
    y = sp.symbols('y')
    J = (y - 5)**2 + 3
    print(f"Função de Custo: J(y) = {J}")

    # Calcular a derivada (Gradiente)
    J_prime = sp.diff(J, y)
    print(f"Derivada (Gradiente): J'(y) = {J_prime}")

    # Encontrar o ponto ótimo analítico (onde J' = 0)
    optimal_y = sp.solve(J_prime, y)[0]
    print(f"Ponto ótimo analítico (J' = 0): y = {optimal_y}")

    # Teste da segunda derivada
    J_double_prime = sp.diff(J_prime, y)
    concavidade = J_double_prime.subs(y, optimal_y)
    print(f"Segunda Derivada J'': {concavidade} ( > 0, logo é um MÍNIMO)")

    # --- Fase 2: Implementação do Algoritmo (NumPy) ---
    print("\nIniciando Algoritmo de Descida do Gradiente (Gradient Descent)...")

    # Converter a derivada simbólica em uma função numérica
    J_prime_func = sp.lambdify(y, J_prime, 'numpy')

    # Parâmetros do algoritmo
    learning_rate = 0.1  # Taxa de aprendizado
    n_iterations = 50    # Número de iterações
    current_y = 15.0     # Ponto de início (arbitrário, longe do ótimo)
    
    history_y = [current_y] # Guarda o histórico de y

    for i in range(n_iterations):
        # Calcula o gradiente no ponto atual
        gradient = J_prime_func(current_y)
        
        # Atualiza o valor de y (dando um passo na direção oposta ao gradiente)
        current_y = current_y - learning_rate * gradient
        
        history_y.append(current_y)
        
        if i % 5 == 0:
             print(f"Iteração {i}: y = {current_y:.4f}, Gradiente = {gradient:.4f}")

    print(f"\nResultado do Gradient Descent após {n_iterations} iterações: y ≈ {current_y:.4f}")
    print(f"Valor ótimo analítico: y = {optimal_y}")

    # --- Visualização ---
    J_func = sp.lambdify(y, J, 'numpy')
    y_vals = np.linspace(-5, 20, 400)
    J_vals = J_func(y_vals)
    
    history_J = J_func(np.array(history_y))

    plt.figure()
    plt.plot(y_vals, J_vals, label=f"J(y) = ${sp.latex(J)}$")
    plt.plot(history_y, history_J, 'ro-', label='Passos do Gradient Descent', markersize=4)
    plt.plot(optimal_y, J_func(optimal_y), 'g*', markersize=15, label=f'Mínimo Analítico (y={optimal_y})')
    
    plt.title("Problema 2: Gradient Descent para Minimizar J(y)")
    plt.xlabel('y')
    plt.ylabel('J(y) (Custo)')
    plt.legend()
    print("Gráfico do Problema 2 gerado.")


def problema_3():
    """
    PROBLEMA 3: Análise de Tendências (Ponto de Inflexão)
    Função: f(t) = -t^3 + 6t^2 + 5
    Objetivo: Encontrar o "ponto de virada" (ponto de inflexão)
    """
    print("\n--- Problema 3: Análise de Tendências (Ponto de Inflexão) ---")

    # 1. Análise Simbólica com SymPy
    t = sp.symbols('t')
    f = -t**3 + 6*t**2 + 5
    print(f"Função (Crescimento): f(t) = {f}")

    # Primeira Derivada (Velocidade do crescimento)
    f_prime = sp.diff(f, t)
    print(f"Primeira Derivada (Velocidade): f'(t) = {f_prime}")

    # Segunda Derivada (Aceleração do crescimento)
    f_double_prime = sp.diff(f_prime, t)
    print(f"Segunda Derivada (Aceleração): f''(t) = {f_double_prime}")

    # Encontrar o Ponto de Inflexão (onde f'' = 0 e muda de sinal)
    # Este é o ponto onde a taxa de crescimento (velocidade) é máxima.
    inflection_points = sp.solve(f_double_prime, t)
    inflection_t = inflection_points[0]
    inflection_f = f.subs(t, inflection_t)
    
    print(f"\nPonto de Inflexão (f'' = 0): t = {inflection_t}")
    print(f"Valor da função no ponto: f({inflection_t}) = {inflection_f.evalf():.4f}")

    # Verificando que a velocidade (f') é máxima neste ponto
    # Encontramos o ponto crítico da velocidade (derivada de f' = f'')
    # Já sabemos que f'' = 0 em t=2.
    # Vamos verificar a concavidade de f' (usando f''')
    f_triple_prime = sp.diff(f_double_prime, t)
    print(f"Teste para f' (f'''): {f_triple_prime}")
    # Como f''' = -6 (negativo), t=2 é um ponto de MÁXIMO para f'(t).
    print(f"Isso confirma que em t={inflection_t}, a velocidade de crescimento f'(t) é MÁXIMA.")

    # 2. Visualização com Matplotlib
    
    # Converter para funções numéricas
    f_func = sp.lambdify(t, f, 'numpy')
    f_prime_func = sp.lambdify(t, f_prime, 'numpy')
    f_double_prime_func = sp.lambdify(t, f_double_prime, 'numpy')
    
    t_vals = np.linspace(-1, 7, 400)
    
    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10)) # 2 linhas, 1 coluna
    
    # Gráfico 1: Função f(t)
    ax1.plot(t_vals, f_func(t_vals), label=f"f(t) = ${sp.latex(f)}$ (Adoção)")
    # Marcar o ponto de inflexão
    ax1.plot(inflection_t, inflection_f, 'ro', markersize=10, label=f'Ponto de Inflexão (t={inflection_t})')
    ax1.set_title('Problema 3: Curva de Adoção de Tecnologia')
    ax1.set_ylabel('f(t) (Usuários)')
    ax1.legend()
    ax1.axvline(x=inflection_t, color='gray', linestyle='--', label=f't={inflection_t}')

    # Gráfico 2: Derivadas f'(t) e f''(t)
    ax2.plot(t_vals, f_prime_func(t_vals), 'g-', label=f"f'(t) = ${sp.latex(f_prime)}$ (Velocidade)")
    ax2.plot(t_vals, f_double_prime_func(t_vals), 'b--', label=f"f''(t) = ${sp.latex(f_double_prime)}$ (Aceleração)")
    
    # Marcar o ponto onde f' é máxima (t=2) e f'' é zero
    max_f_prime = f_prime_func(inflection_t)
    ax2.plot(inflection_t, max_f_prime, 'go', markersize=10, label=f"Velocidade Máxima (t={inflection_t})")
    ax2.plot(inflection_t, 0, 'bo', markersize=10, label=f"Aceleração Nula (t={inflection_t})")
    
    ax2.set_ylabel('Taxa de Variação')
    ax2.set_xlabel('t (Tempo)')
    ax2.legend()
    ax2.axvline(x=inflection_t, color='gray', linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    print("Gráfico do Problema 3 gerado.")


def problema_4():
    """
    PROBLEMA 4: Caso Prático (Otimização de Recurso)
    Problema: Maximizar a área de um terreno retangular que pode ser
    cercado com 200 metros de cerca.
    """
    print("\n--- Problema 4: Caso Prático (Maximizar Área) ---")
    
    # 1. Modelagem Matemática
    # Perímetro: 2L + 2W = 200  =>  L + W = 100  =>  L = 100 - W
    # Área: A = L * W = (100 - W) * W = 100W - W^2
    
    # 2. Análise Simbólica com SymPy
    W = sp.symbols('W')
    A = 100*W - W**2
    print(f"Função da Área: A(W) = {A}")

    # Primeira Derivada (Taxa de variação da Área)
    A_prime = sp.diff(A, W)
    print(f"Primeira Derivada: A'(W) = {A_prime}")

    # Ponto crítico (onde A' = 0)
    critical_W = sp.solve(A_prime, W)[0]
    print(f"Ponto crítico (A' = 0): W = {critical_W} metros")

    # Calcular L
    L = 100 - critical_W
    print(f"Dimensões para área máxima: L = {L} metros, W = {critical_W} metros")

    # Teste da segunda derivada
    A_double_prime = sp.diff(A_prime, W)
    print(f"Segunda Derivada: A''(W) = {A_double_prime}")
    # Como A'' = -2 (negativo), o ponto é um MÁXIMO LOCAL.
    print("Como A'' < 0, o ponto é um MÁXIMO LOCAL.")
    
    max_area = A.subs(W, critical_W)
    print(f"Área Máxima: {max_area} m²")

    # 3. Visualização
    A_func = sp.lambdify(W, A, 'numpy')
    W_vals = np.linspace(0, 100, 400) # W pode ir de 0 a 100
    A_vals = A_func(W_vals)
    
    plt.figure()
    plt.plot(W_vals, A_vals, label=f"A(W) = ${sp.latex(A)}$")
    plt.plot(critical_W, max_area, 'ro', label=f'Área Máxima (W={critical_W}m, A={max_area}m²)')
    
    plt.title('Problema 4: Maximização de Área com Perímetro Fixo')
    plt.xlabel('W (Largura em metros)')
    plt.ylabel('A(W) (Área em m²)')
    plt.legend()
    print("Gráfico do Problema 4 gerado.")


def problema_5():
    """
    PROBLEMA 5: Caso Prático (Análise de Movimento - Física/Robótica)
    Problema: A posição de um robô é dada por s(t) = t³ - 6t² + 9t + 1,
    onde 's' é a distância em metros e 't' é o tempo em segundos.
    a) Encontre a velocidade v(t) e a aceleração a(t).
    b) Quando o robô para (velocidade = 0)?
    c) Quando a aceleração é zero? Qual a velocidade nesse instante?
    """
    print("\n--- Problema 5: Caso Prático (Análise de Movimento) ---")

    # 1. Análise Simbólica com SymPy
    t = sp.symbols('t')
    s = t**3 - 6*t**2 + 9*t + 1
    print(f"Função Posição: s(t) = {s}")
    
    # a) Encontrar velocidade v(t) e aceleração a(t)
    # Velocidade é a primeira derivada da posição
    v = sp.diff(s, t)
    print(f"Função Velocidade: v(t) = s'(t) = {v}")
    
    # Aceleração é a segunda derivada da posição (ou primeira da velocidade)
    a = sp.diff(v, t)
    print(f"Função Aceleração: a(t) = s''(t) = {a}")
    
    # b) Quando o robô para (v(t) = 0)?
    t_parado = sp.solve(v, t)
    print(f"\nRobô para (v=0) nos instantes t = {t_parado} segundos.")
    
    # c) Quando a aceleração é zero (ponto de inflexão da posição)?
    t_acel_zero = sp.solve(a, t)
    t_inflexao = t_acel_zero[0]
    print(f"\nAceleração é zero (a=0) no instante t = {t_inflexao} segundos.")
    
    # Calcular a velocidade nesse instante
    v_no_ponto = v.subs(t, t_inflexao)
    print(f"Velocidade no instante t={t_inflexao}: v({t_inflexao}) = {v_no_ponto} m/s")
    # Este é o instante de velocidade mínima (mais negativa).

    # 2. Visualização
    s_func = sp.lambdify(t, s, 'numpy')
    v_func = sp.lambdify(t, v, 'numpy')
    a_func = sp.lambdify(t, a, 'numpy')
    
    t_vals = np.linspace(0, 5, 400)
    
    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    
    # Gráfico 1: Posição s(t)
    ax1.plot(t_vals, s_func(t_vals), label=f"s(t) = ${sp.latex(s)}$ (Posição)")
    # Marcar pontos onde v=0
    for t_val in t_parado:
        ax1.plot(t_val, s_func(t_val), 'ro', label=f'Parada (t={t_val}s)')
    # Marcar ponto de inflexão (a=0)
    ax1.plot(t_inflexao, s_func(t_inflexao), 'go', label=f'Aceleração Zero (t={t_inflexao}s)')
    ax1.set_title('Problema 5: Análise de Movimento do Robô')
    ax1.set_ylabel('s(t) (Posição em m)')
    ax1.legend()
    ax1.axvline(x=t_inflexao, color='gray', linestyle='--')

    # Gráfico 2: Velocidade v(t) e Aceleração a(t)
    ax2.plot(t_vals, v_func(t_vals), 'g-', label=f"v(t) = ${sp.latex(v)}$ (Velocidade)")
    ax2.plot(t_vals, a_func(t_vals), 'b--', label=f"a(t) = ${sp.latex(a)}$ (Aceleração)")
    # Marcar pontos onde v=0
    for t_val in t_parado:
        ax2.plot(t_val, 0, 'ro')
    # Marcar ponto onde a=0
    ax2.plot(t_inflexao, a_func(t_inflexao), 'go')
    ax2.plot(t_inflexao, v_func(t_inflexao), 'g*', markersize=10, label=f'Velocidade Mínima (t={t_inflexao}s)')
    
    ax2.set_ylabel('Velocidade (m/s) / Aceleração (m/s²)')
    ax2.set_xlabel('t (Tempo em segundos)')
    ax2.legend()
    ax2.axvline(x=t_inflexao, color='gray', linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    print("Gráfico do Problema 5 gerado.")


def main():
    """
    Função principal para executar todas as análises dos problemas.
    """
    problema_1()
    problema_2()
    problema_3()
    problema_4()
    problema_5()
    
    print("\n--- Execução Concluída ---")
    print("Exibindo todos os gráficos...")
    plt.show() # Exibe todas as figuras criadas


if __name__ == "__main__":
    main()