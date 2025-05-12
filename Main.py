import pandas as pd
import numpy as np
import time
from itertools import combinations, chain
from collections import defaultdict

# 1. UNIFICAÇÃO DOS DADOS

try:
    t1 = time.time()
    print("🔄 Carregando os dados...")
    ratings = pd.read_csv("data/ratings_small.csv", low_memory=False).head(3000) # Ajustar conforme demanda para teste
    movies = pd.read_csv("data/movies_metadata.csv", low_memory=False).head(3000) # Ajustar conforme demanda para teste

    print("🧼 Corrigindo tipos e removendo IDs inválidos...")
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
    movies.dropna(subset=['id'], inplace=True)
    ratings.dropna(subset=['movieId'], inplace=True)

    print("🔗 Unificando os dados...")
    movies = movies.rename(columns={"id": "movieId"})
    merged = pd.merge(ratings, movies, on="movieId")
    t2 = time.time()
    print(f"✅ Etapa 1 concluída em {t2 - t1:.2f} segundos!\n")
except Exception as e:
    print(f"❌ Erro na etapa 1 (Unificação dos Dados): {e}")

# 2. PRÉ-PROCESSAMENTO

try:
    t2 = time.time()
    print("🔍 Filtrando avaliações maiores que 3...")
    filtered = merged[merged['rating'] > 3]

    print("👥 Agrupando filmes por usuário...")
    user_movies = filtered.groupby('userId')['title'].apply(list)
    t3 = time.time()
    print(f"✅ Etapa 2 concluída em {t3 - t2:.2f} segundos!\n")
except Exception as e:
    print(f"❌ Erro na etapa 2 (Pré-processamento): {e}")



# 3. IMPLEMENTAÇÃO DO APRIORI

def get_frequent_itemsets(transactions, min_support):
    itemsets = defaultdict(int)
    for transaction in transactions:
        for item in set(transaction):
            itemsets[frozenset([item])] += 1

    total = len(transactions)
    frequent = {item: count / total for item, count in itemsets.items() if count / total >= min_support}
    result = dict(frequent)
    k = 2
    current_itemsets = list(frequent.keys())

    while current_itemsets:
        unique_items = set(chain.from_iterable(current_itemsets))
        candidates = list(combinations(unique_items, k))
        candidate_counts = defaultdict(int)

        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if set(candidate).issubset(transaction_set):
                    candidate_counts[frozenset(candidate)] += 1

        frequent = {
            item: count / total
            for item, count in candidate_counts.items()
            if count / total >= min_support
        }

        if not frequent:
            break

        result.update(frequent)
        current_itemsets = list(frequent.keys())
        k += 1

    return result

def generate_rules(frequent_itemsets, min_confidence=0.4, min_lift=0.01):  # Ajustar conforme demanda para teste
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(itemset, antecedent_size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                support_itemset = frequent_itemsets[itemset]
                support_antecedent = frequent_itemsets.get(antecedent, 0)
                support_consequent = frequent_itemsets.get(consequent, 0)

                if support_antecedent == 0:
                    continue

                confidence = support_itemset / support_antecedent
                lift = confidence / support_consequent if support_consequent > 0 else 0

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": set(antecedent),
                        "consequent": set(consequent),
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift
                    })
    return rules

try:
    t3 = time.time()
    print("⚙️ Gerando itemsets frequentes...")
    transactions = user_movies.tolist()

    if not transactions:
        raise ValueError("Não há transações para gerar itemsets.")

    min_support = 0.1  # Ajustar conforme demanda para teste
    frequent_itemsets = get_frequent_itemsets(transactions, min_support)

    if not frequent_itemsets:
        raise ValueError("Nenhum itemset frequente foi encontrado. Reduza o valor de min_support.")

    print("📊 Gerando regras de associação...")
    rules = generate_rules(frequent_itemsets, min_confidence=0.5, min_lift=0.01) # Ajustar conforme demanda para teste, o lift pode interferir caso esteja mais alto

    if not rules:
        raise ValueError("Nenhuma regra de associação foi gerada. Ajuste os parâmetros.")

    print(f"Total de regras geradas: {len(rules)}")
    t4 = time.time()
    print(f"✅ Etapa 3 concluída em {t4 - t3:.2f} segundos!\n")
except Exception as e:
    print(f"❌ Erro na etapa 3 (Apriori): {e}")

# 4. SISTEMA DE RECOMENDAÇÃO

def recommend_movies(user_liked_movies, rules):
    recommendations = set()
    for rule in rules:
        if rule['antecedent'].issubset(user_liked_movies):
            recommendations.update(rule['consequent'])
    return recommendations - user_liked_movies

try:
    t4 = time.time()
    print("🎯 Gerando recomendações para um usuário de exemplo...")

    usuario_exemplo = set(user_movies.iloc[0])  # Garantir que tem filmes
    if not usuario_exemplo:
        raise ValueError("O usuário de exemplo não tem filmes para recomendar.")

    sugestoes = recommend_movies(usuario_exemplo, rules)

    if not sugestoes:
        raise ValueError("Nenhuma sugestão de filme gerada para o usuário de exemplo.")

    print("\n🎬 Filmes assistidos e gostados pelo usuário:")
    print(usuario_exemplo)
    print("\n📽️ Sugestões de filmes com base nas regras de associação:")
    print(sugestoes)

    t5 = time.time()
    print(f"\n✅ Etapa 4 concluída em {t5 - t4:.2f} segundos!")
except Exception as e:
    print(f"❌ Erro na etapa 4 (Sistema de Recomendação): {e}")


# 5. SALVAMENTO DAS REGRAS

try:
    print("\n💾 Salvando as regras em arquivo Excel...")

    # Salvar como Excel
    df_rules = pd.DataFrame(rules)
    df_rules.to_excel("regras_associacao.xlsx", index=False)

    print("✅ Regras salvas com sucesso em 'regras_associacao.xlsx'")
except Exception as e:
    print(f"❌ Erro ao salvar as regras: {e}")

# 6. SALVAMENTO DAS RECOMENDAÇÕES

try:
    print("\n💾 Salvando as recomendações do usuário exemplo em Excel...")

    # Salvar como Excel
    pd.DataFrame({"recomendacoes": list(sugestoes)}).to_excel("recomendacoes_usuario.xlsx", index=False)

    print("✅ Recomendações salvas com sucesso em 'recomendacoes_usuario.xlsx'")
except Exception as e:
    print(f"❌ Erro ao salvar as recomendações: {e}")
