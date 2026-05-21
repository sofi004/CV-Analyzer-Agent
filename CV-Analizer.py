import os # Permite interagir com o sistema de ficheiros do sistema operativo
import glob # Usado para procurar caminhos de ficheiros que correspondam a um padrão específico (.pdf)
import pandas as pd # Biblioteca essencial para manipulação de dados estruturados em tabelas
import gradio as gr # Framework para construir a interface gráfica web
from pypdf import PdfReader # Extrai o texto contido dentro de documentos PDF
from sklearn.metrics.pairwise import cosine_similarity # Função matemática que calcula o quão próximos dois vetores embeddings estão um do outro
from sentence_transformers import SentenceTransformer # Carrega modelos open-source locais que convertem texto em vetores de significado embeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Classe da LangChain que faz a ponte de comunicação com os modelos LLM da Google (Gemini)
# CONFIGURAÇÃO DO LLM E EMBEDDINGS ---
# REWRITE THIS FUNCTION TO LOOK LIKE THIS:
def config_llm_gemini():
    # Puxa de forma segura a chave das variáveis de ambiente do Ubuntu
    api_key = os.environ.get('GOOGLE_API_KEY') 

    if not api_key:
        raise ValueError("ERRO: A variável de ambiente GOOGLE_API_KEY não foi encontrada! "
                         "Corre o comando 'export GOOGLE_API_KEY=tuachave' no teu terminal.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0 
    )

# Modelo para transformar texto em vetores numéricos
embedder = SentenceTransformer('all-MiniLM-L6-v2') # Descarrega e inicializa o modelo all-MiniLM-L6-v2 na memória do Colab. Ele será o responsável por processar semanticamente o texto dos CVs e JDs localmente
llm = config_llm_gemini() # Ativa a instância do Gemini guardando-a na variável llm para uso posterior

# FUNÇÕES DE PROCESSAMENTO ---
def extract_text_from_pdf(pdf_path):
    try: # O bloco try/except garante que se um PDF estiver corrompido, o script não crasha e devolve apenas uma string vazia
        reader = PdfReader(pdf_path)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()]) # Usa uma compreensão de lista para iterar por todas as páginas do documento, extrair o texto de cada uma delas e juntar tudo numa única string longa.
    except:
        return ""

def get_match_score(cv_text, jd_text):
    # Transforma textos em embeddings e calcula similaridade
    embeddings = embedder.encode([cv_text, jd_text])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def get_ai_justification(candidate_name, cv_text, jd_text):
    # O LLM entra aqui para dar o "Consulting Mindset" da EY
    prompt = f"""
    Como consultor de RH da EY, justifica em 2 frases curtas porque é que o candidato {candidate_name}
    é um bom match para esta vaga, focando em competências técnicas específicas.
    Vaga: {jd_text[:500]}...
    CV: {cv_text[:1000]}...
    """
    # Prompt do agente injeta o nome do candidato, os primeiros 500 caracteres da vaga e os primeiros 1000 caracteres do CV para evitar passar do limite de tokens desnecessários e focar no topo do CV onde costuma estar o resumo executivo
    response = llm.invoke(prompt) # Envia o comando para os servidores da Google e recebe a resposta estruturada
    return response.content

# CARREGAMENTO DE DADOS DASHBOARD PREP
cv_files = glob.glob("CVs/*.pdf")
jd_files = glob.glob("JobDescriptions/*.pdf")
# glob.glob(...): Mapeia e cria listas contendo os caminhos de todos os ficheiros .pdf encontrados dentro da pasta CVs e da pasta JobDescriptions

print(f"A carregar {len(cv_files)} CVs...")
cv_database = [] # Cria uma lista vazia cv_database
for path in cv_files:
    text = extract_text_from_pdf(path)
    if text: cv_database.append({"name": path.split("/")[-1], "text": text})
# Itera por cada caminho de CV encontrado, extrai o texto completo do PDF e adiciona um dicionário à lista com o nome limpo do ficheiro (path.split("/")[-1], que remove o caminho das pastas deixando apenas o seu conteúdo textual.

print(f"A carregar {len(jd_files)} Vagas...")
jd_database = {path.split("/")[-1].replace(".pdf", ""): extract_text_from_pdf(path) for path in jd_files}
# Cria um dicionário (através de uma compreensão de dicionário) onde a chave é o nome do ficheiro da vaga sem o .pdf e o valor é o texto integral extraído desse documento.

# LÓGICA DA INTERFACE (RANKING + AGENTE)
def final_agent_analysis(job_title):
    if not job_title: 
        return pd.DataFrame(columns=["Candidato", "Match %"]), "Por favor, selecione uma vaga no menu acima antes de analisar."

    jd_text = jd_database[job_title]
    results = []

    # Calcular scores para todos (Matemático)
    for cv in cv_database:
        score = get_match_score(cv['text'], jd_text)
        results.append({"Candidato": cv['name'], "score": score, "full_text": cv['text']})
    # Percorre os 107 candidatos já carregados em memória, calcula o score de cosseno contra a vaga atual e adiciona os resultados a uma lista temporária.

    # Rankear e pegar o Top 3 para análise profunda do LLM
    df = pd.DataFrame(results).sort_values("score", ascending=False).head(3)
    # Converte a lista de resultados num DataFrame do Pandas.
    # .sort_values("score", ascending=False): Ordena a tabela do maior score para o menor.
    #.head(3): Isola apenas as 3 melhores linhas (o Top 3 de candidatos ideais).

    # O Gemini analisa o Top 1 para dar o insight final
    top_1_cv = df.iloc[0]
    justification = get_ai_justification(top_1_cv['Candidato'], top_1_cv['full_text'], jd_text)
    # Envia os dados desse vencedor para a função que liga para o Gemini para gerar a análise em linguagem natural de forma cirúrgica.

    # Formatar output para a tabela
    final_df = df[["Candidato", "score"]].copy() # Gera uma cópia limpa da tabela contendo apenas as colunas necessárias.
    final_df["Match %"] = (final_df["score"] * 100).round(2) # Multiplica o score por 100 e arredonda a duas casas decimais para transformar o valor matemático puro numa percentagem amigável para o negócio

    return final_df[["Candidato", "Match %"]], f"Recomendação do Agente para {top_1_cv['Candidato']}:\n{justification}"
    # Retorna dois elementos: o DataFrame estruturado (para a tabela) e a string com o parecer técnico gerado pelo Gemini.

import gradio as gr

# --- CONFIGURAÇÃO DE CORES EY ---
# Amarelo EY: #FFE600
# Preto: #000000
# Branco: #FFFFFF

ey_theme = gr.themes.Soft(
    primary_hue="yellow",      # Define a cor dos botões principais
    secondary_hue="gray",
    neutral_hue="gray",
).set(
    button_primary_background_fill="#FFE600",
    button_primary_background_fill_hover="#E6CF00",
    button_primary_text_color="#000000",
    body_background_fill="#FFFFFF",
    block_title_text_size="18px",
    block_title_text_color="#000000"
)

# CSS Customizado para garantir que o cabeçalho e elementos específicos sigam a paleta
custom_css = """
footer {display: none !important;}
.gradio-container {background-color: #FFFFFF;}
h1 {color: #000000 !important; font-weight: 800 !important;}
#ey-header {background-color: #000000; padding: 20px; border-radius: 8px; margin-bottom: 20px;}
#ey-header h1, #ey-header h3 {color: #FFE600 !important; margin: 0;}
"""

with gr.Blocks(theme=ey_theme, css=custom_css) as demo:
    # Cabeçalho estilizado com fundo preto e fonte amarela
    with gr.Column(elem_id="ey-header"):
        gr.Markdown("# EY AI Challenge - CV Analyzer Agent")
        gr.Markdown("### Seleção Estratégica baseada em Similaridade Semântica e IA Generativa")

    with gr.Row():
        vaga_input = gr.Dropdown(
            choices=list(jd_database.keys()),
            label="Posição Crítica EY",
            info="Selecione a vaga para iniciar o screening"
        )
        btn = gr.Button("Analisar Candidatos", variant="primary")

    with gr.Column():
        tabela_output = gr.Dataframe(
            label="Top Candidatos (Cosine Similarity)",
            interactive=False
        )
        insight_output = gr.Textbox(
            label="Análise do Consultor AI (Gemini Flash)",
            lines=4,
            placeholder="Os insights detalhados aparecerão aqui após a análise..."
        )

    btn.click(
        fn=final_agent_analysis,
        inputs=vaga_input,
        outputs=[tabela_output, insight_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)