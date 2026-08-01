import json
import logging
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CẤU HÌNH
# ============================================================

st.set_page_config(
    page_title="TaxRAG VN",
    page_icon="📘",
    layout="wide",
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("taxrag")

DATA_DIR = Path("./data_store")
DATA_FILE = DATA_DIR / "chunks.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_setting(name: str, default: str = "") -> str:
    """Đọc cấu hình từ Streamlit Secrets hoặc biến môi trường."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.6-flash")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


gemini_client = get_gemini_client()


# ============================================================
# 2. LƯU VÀ ĐỌC KHO DỮ LIỆU
# ============================================================

def load_chunks() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []

    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("Không thể đọc kho dữ liệu")
        return []


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    temp_file = DATA_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(DATA_FILE)


def delete_all_chunks() -> None:
    save_chunks([])


# ============================================================
# 3. XỬ LÝ PDF
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return text.strip()


def split_text(
    text: str,
    max_chars: int = 1400,
    overlap_chars: int = 220,
) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip()

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        temp = ""

        for sentence in sentences:
            candidate_sentence = f"{temp} {sentence}".strip()

            if len(candidate_sentence) <= max_chars:
                temp = candidate_sentence
            else:
                if temp:
                    chunks.append(temp)
                temp = sentence

        current = temp

    if current:
        chunks.append(current)

    overlapped: List[str] = []
    previous_tail = ""

    for chunk in chunks:
        merged = (
            f"{previous_tail}\n{chunk}".strip()
            if previous_tail
            else chunk
        )
        overlapped.append(merged)
        previous_tail = chunk[-overlap_chars:]

    return overlapped


def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_index, page in enumerate(doc):
                pages.append(
                    {
                        "page": page_index + 1,
                        "text": clean_text(page.get_text("text")),
                    }
                )
    except Exception as exc:
        raise ValueError(
            "Không thể đọc PDF. File có thể bị lỗi hoặc được bảo vệ."
        ) from exc

    return pages


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    document_name: str,
    document_number: str,
    topic: str,
    status: str,
    source_url: str,
) -> Dict[str, int]:
    pages = extract_pdf_pages(pdf_bytes)
    document_key = document_number.strip() or filename

    existing = [
        item
        for item in load_chunks()
        if item.get("document_key") != document_key
    ]

    new_chunks: List[Dict[str, Any]] = []
    empty_pages = 0

    for page_info in pages:
        page_number = int(page_info["page"])
        page_text = str(page_info["text"])

        if len(page_text) < 50:
            empty_pages += 1
            continue

        for chunk_index, content in enumerate(split_text(page_text)):
            chunk_id = sha256(
                f"{document_key}|{page_number}|{chunk_index}".encode("utf-8")
            ).hexdigest()

            new_chunks.append(
                {
                    "id": chunk_id,
                    "document_key": document_key,
                    "filename": filename,
                    "document_name": document_name,
                    "document_number": document_number,
                    "topic": topic,
                    "status": status,
                    "page": page_number,
                    "source_url": source_url,
                    "content": content,
                }
            )

    save_chunks(existing + new_chunks)

    return {
        "pages": len(pages),
        "chunks": len(new_chunks),
        "empty_pages": empty_pages,
    }


# ============================================================
# 4. TRUY XUẤT TF-IDF
# ============================================================

def retrieve_context(
    question: str,
    top_k: int = 5,
    active_only: bool = True,
) -> List[Dict[str, Any]]:
    chunks = load_chunks()

    if active_only:
        chunks = [
            item
            for item in chunks
            if item.get("status") == "Còn hiệu lực"
        ]

    if not chunks:
        return []

    documents = [
        (
            f"{item.get('document_name', '')} "
            f"{item.get('document_number', '')} "
            f"{item.get('topic', '')} "
            f"{item.get('content', '')}"
        )
        for item in chunks
    ]

    # Kết hợp đặc trưng từ và ký tự để xử lý tiếng Việt, lỗi gõ và viết tắt.
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    word_matrix = word_vectorizer.fit_transform(documents)
    char_matrix = char_vectorizer.fit_transform(documents)

    doc_matrix = hstack([word_matrix, char_matrix])

    query_word = word_vectorizer.transform([question])
    query_char = char_vectorizer.transform([question])
    query_matrix = hstack([query_word, query_char])

    scores = cosine_similarity(query_matrix, doc_matrix).flatten()
    safe_top_k = min(max(1, top_k), len(chunks))
    best_indices = np.argsort(scores)[::-1][:safe_top_k]

    results: List[Dict[str, Any]] = []

    for index in best_indices:
        item = dict(chunks[int(index)])
        item["similarity"] = float(scores[int(index)])
        results.append(item)

    return results


SYSTEM_PROMPT = """
Bạn là trợ lý tra cứu chính sách thuế Việt Nam dành riêng cho doanh nghiệp vừa và nhỏ.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp.
2. Không tự tạo số hiệu văn bản, điều, khoản, ngày tháng, mức thuế hoặc thời hạn.
3. Khi ngữ cảnh không đủ, phải nói rõ:
   "Chưa đủ căn cứ trong kho dữ liệu để kết luận."
4. Diễn đạt đơn giản, thực tế và phù hợp với doanh nghiệp vừa và nhỏ.
5. Mỗi nhận định pháp lý quan trọng phải gắn với [Nguồn 1], [Nguồn 2]...
6. Khi nguồn mâu thuẫn hoặc khác thời điểm áp dụng, phải nêu rõ.
7. Không hướng dẫn trốn thuế, che giấu doanh thu hoặc vi phạm pháp luật.
8. Không thay thế tư vấn chính thức của cơ quan thuế hoặc chuyên gia.

CẤU TRÚC:
- Kết luận ngắn gọn
- Giải thích
- Căn cứ được sử dụng
- Lưu ý hoặc ngoại lệ
"""


def build_context(retrieved_items: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []

    for index, item in enumerate(retrieved_items, start=1):
        blocks.append(
            f"""
[NGUỒN {index}]
Tên văn bản: {item.get('document_name', '')}
Số hiệu: {item.get('document_number', '')}
Chủ đề: {item.get('topic', '')}
Trạng thái: {item.get('status', '')}
Trang: {item.get('page', '')}
URL: {item.get('source_url', '')}

Nội dung:
{item.get('content', '')}
""".strip()
        )

    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    retrieved_items: List[Dict[str, Any]],
) -> str:
    if gemini_client is None:
        return (
            "Hệ thống chưa được cấu hình GEMINI_API_KEY. "
            "Quản trị viên cần kiểm tra Streamlit Secrets."
        )

    prompt = f"""
{SYSTEM_PROMPT}

CÂU HỎI:
{question}

NGỮ CẢNH:
{build_context(retrieved_items)}

Hãy trả lời bằng tiếng Việt và không dùng kiến thức ngoài NGỮ CẢNH.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = getattr(response, "text", None)
        return answer or "Mô hình không trả về nội dung."

    except Exception:
        logger.exception("Không thể gọi Gemini API")
        return (
            "Hệ thống tạm thời không thể tạo câu trả lời. "
            "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        )


# ============================================================
# 5. GIAO DIỆN
# ============================================================

st.title("TaxRAG VN")
st.caption(
    "Trợ lý tra cứu chính sách thuế Việt Nam dành cho doanh nghiệp vừa và nhỏ."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_admin, tab_guide = st.tabs(
    ["Hỏi đáp", "Quản trị dữ liệu", "Hướng dẫn"]
)


with tab_chat:
    col_filter, col_chat = st.columns([1, 3])

    with col_filter:
        st.subheader("Bộ lọc")

        active_only = st.checkbox(
            "Chỉ dùng văn bản còn hiệu lực",
            value=True,
        )

        top_k = st.slider(
            "Số đoạn truy xuất",
            min_value=3,
            max_value=10,
            value=5,
        )

        min_similarity = st.slider(
            "Ngưỡng liên quan tối thiểu",
            min_value=0.00,
            max_value=0.80,
            value=0.08,
            step=0.01,
        )

        st.metric("Số đoạn trong kho", len(load_chunks()))
        st.caption(f"Model Gemini: `{GEMINI_MODEL}`")

    with col_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("Nguồn được truy xuất"):
                        for source in message["sources"]:
                            st.markdown(source)

        question = st.chat_input(
            "Ví dụ: Doanh nghiệp mới thành lập chưa có doanh thu "
            "có phải khai thuế GTGT không?"
        )

        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                retrieved_items = retrieve_context(
                    question=question,
                    top_k=top_k,
                    active_only=active_only,
                )

                if not retrieved_items:
                    answer = (
                        "Kho dữ liệu đang trống hoặc không có văn bản "
                        "phù hợp với bộ lọc hiện tại."
                    )
                    st.warning(answer)
                else:
                    best_similarity = retrieved_items[0]["similarity"]

                    if best_similarity < min_similarity:
                        answer = (
                            "Chưa đủ căn cứ trong kho dữ liệu để kết luận. "
                            "Hãy diễn đạt câu hỏi cụ thể hơn hoặc bổ sung tài liệu."
                        )
                        st.warning(answer)
                    else:
                        with st.spinner(
                            "Đang tìm căn cứ và tạo câu trả lời..."
                        ):
                            answer = generate_answer(
                                question=question,
                                retrieved_items=retrieved_items,
                            )
                        st.markdown(answer)

                source_lines: List[str] = []

                for index, item in enumerate(retrieved_items, start=1):
                    line = (
                        f"**Nguồn {index}:** "
                        f"{item.get('document_name', '')} — "
                        f"{item.get('document_number', '')} — "
                        f"trang {item.get('page', '')} — "
                        f"độ tương đồng {item.get('similarity', 0):.2f}"
                    )

                    if item.get("source_url"):
                        line += f" — [Mở nguồn]({item['source_url']})"

                    source_lines.append(line)

                if source_lines:
                    with st.expander("Nguồn được truy xuất"):
                        for line in source_lines:
                            st.markdown(line)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_lines,
                }
            )


with tab_admin:
    st.subheader("Quản trị dữ liệu")

    admin_allowed = True

    if ADMIN_PASSWORD:
        entered_password = st.text_input(
            "Mật khẩu quản trị",
            type="password",
        )
        admin_allowed = entered_password == ADMIN_PASSWORD

        if entered_password and not admin_allowed:
            st.error("Mật khẩu không đúng.")

    else:
        st.warning(
            "Chưa cấu hình ADMIN_PASSWORD. "
            "Tab quản trị hiện chưa được bảo vệ."
        )

    if admin_allowed:
        uploaded_file = st.file_uploader(
            "Chọn một file PDF có lớp chữ",
            type=["pdf"],
        )

        document_name = st.text_input(
            "Tên văn bản",
            placeholder="Ví dụ: Nghị định quy định về hóa đơn, chứng từ",
        )

        document_number = st.text_input(
            "Số hiệu văn bản",
            placeholder="Ví dụ: 123/2020/NĐ-CP",
        )

        topic = st.selectbox(
            "Chủ đề",
            [
                "Thuế giá trị gia tăng",
                "Hóa đơn điện tử",
                "Thủ tục khai và nộp thuế",
            ],
        )

        status = st.selectbox(
            "Trạng thái hiệu lực",
            [
                "Còn hiệu lực",
                "Hết hiệu lực",
                "Chưa xác minh",
            ],
        )

        source_url = st.text_input(
            "Đường dẫn nguồn chính thức",
            placeholder="https://...",
        )

        if st.button("Xử lý và lưu vào kho", type="primary"):
            if uploaded_file is None:
                st.error("Bạn chưa chọn file PDF.")
            elif not document_name.strip():
                st.error("Bạn chưa nhập tên văn bản.")
            elif not document_number.strip():
                st.error("Bạn chưa nhập số hiệu văn bản.")
            else:
                try:
                    with st.spinner("Đang xử lý PDF..."):
                        result = ingest_pdf(
                            pdf_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            document_name=document_name.strip(),
                            document_number=document_number.strip(),
                            topic=topic,
                            status=status,
                            source_url=source_url.strip(),
                        )

                    if result["chunks"] == 0:
                        st.error(
                            "Không trích xuất được chữ. "
                            "File có thể là PDF scan."
                        )
                    else:
                        st.success(
                            f"Đã xử lý {result['pages']} trang và tạo "
                            f"{result['chunks']} đoạn dữ liệu."
                        )

                        if result["empty_pages"] > 0:
                            st.warning(
                                f"Có {result['empty_pages']} trang "
                                "không trích xuất được chữ."
                            )

                except Exception:
                    logger.exception("Không thể xử lý PDF")
                    st.error(
                        "Không thể xử lý tài liệu. "
                        "Hãy kiểm tra file PDF và thử lại."
                    )

        st.divider()

        confirm_delete = st.checkbox(
            "Tôi xác nhận muốn xóa toàn bộ kho dữ liệu"
        )

        if st.button(
            "Xóa toàn bộ dữ liệu",
            disabled=not confirm_delete,
        ):
            delete_all_chunks()
            st.session_state.messages = []
            st.success("Đã xóa toàn bộ dữ liệu.")


with tab_guide:
    st.markdown(
        """
### Phạm vi MVP

Hệ thống chỉ phục vụ **doanh nghiệp vừa và nhỏ**, tập trung vào:

1. Thuế giá trị gia tăng.
2. Hóa đơn điện tử.
3. Thủ tục khai và nộp thuế.

### Cách sử dụng

1. Mở tab **Quản trị dữ liệu**.
2. Tải PDF có thể bôi đen và sao chép chữ.
3. Nhập metadata và lưu tài liệu.
4. Quay lại tab **Hỏi đáp**.
5. Kiểm tra câu trả lời và nguồn được truy xuất.

### Giới hạn

- MVP đang dùng truy xuất TF-IDF để triển khai ổn định, nhẹ.
- Chưa xử lý PDF scan và OCR.
- Chưa có semantic embedding và reranking.
- Dữ liệu cục bộ có thể mất khi Streamlit Cloud redeploy.
- Không thay thế tư vấn chính thức của cơ quan thuế.
"""
    )
