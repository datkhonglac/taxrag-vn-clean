import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CẤU HÌNH ỨNG DỤNG
# ============================================================

# Đây phải là lệnh Streamlit đầu tiên trong chương trình.
st.set_page_config(
    page_title="TaxRAG VN - MVP",
    page_icon="📘",
    layout="wide",
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("taxrag")


def get_setting(name: str, default: str = "") -> str:
    """
    Ưu tiên đọc cấu hình từ Streamlit Secrets.
    Nếu không có, đọc từ biến môi trường của Codespaces hoặc máy cá nhân.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return os.getenv(name, default)


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.6-flash")
CHROMA_DIR = get_setting("CHROMA_DIR", "./database/chroma")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")
COLLECTION_NAME = "tax_policy_vn_sme"

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. KHỞI TẠO MÔ HÌNH VÀ CƠ SỞ DỮ LIỆU
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """Tải mô hình embedding đa ngôn ngữ và giữ trong cache."""
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """Khởi tạo ChromaDB cục bộ và lấy collection của dự án."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Khởi tạo Gemini client nếu API key đã được cấu hình."""
    if not GEMINI_API_KEY:
        return None

    return genai.Client(api_key=GEMINI_API_KEY)


try:
    embedding_model = load_embedding_model()
    collection = get_chroma_collection()
    gemini_client = get_gemini_client()
except Exception:
    logger.exception("Không thể khởi tạo hệ thống")
    st.error(
        "Không thể khởi tạo mô hình embedding hoặc cơ sở dữ liệu. "
        "Hãy kiểm tra requirements.txt và log triển khai."
    )
    st.stop()


# ============================================================
# 3. HÀM XỬ LÝ TÀI LIỆU
# ============================================================

def clean_text(text: str) -> str:
    """
    Làm sạch cơ bản nhưng không thay đổi số liệu, điều, khoản,
    ngày tháng hoặc ký hiệu pháp lý.
    """
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
    """
    Chia nội dung theo đoạn văn; nếu đoạn quá dài thì chia tiếp theo câu.
    Có overlap để hạn chế mất ngữ cảnh giữa hai chunk.
    """
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

        if len(paragraph) > max_chars:
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
        else:
            current = paragraph

    if current:
        chunks.append(current)

    overlapped_chunks: List[str] = []
    previous_tail = ""

    for chunk in chunks:
        merged = (
            f"{previous_tail}\n{chunk}".strip()
            if previous_tail
            else chunk
        )

        overlapped_chunks.append(merged)
        previous_tail = chunk[-overlap_chars:]

    return overlapped_chunks


def extract_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Trích xuất chữ theo từng trang.
    PDF scan không có lớp chữ sẽ cho nội dung ngắn hoặc rỗng.
    """
    pages: List[Tuple[int, str]] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_index, page in enumerate(doc):
                text = clean_text(page.get_text("text"))
                pages.append((page_index + 1, text))

    except Exception as exc:
        raise ValueError(
            "Không thể đọc file PDF. File có thể bị lỗi hoặc được bảo vệ."
        ) from exc

    return pages


def make_chunk_id(
    document_key: str,
    page: int,
    chunk_index: int,
) -> str:
    """
    Tạo ID ổn định. Khi tải lại cùng văn bản, ChromaDB sẽ cập nhật
    dữ liệu thay vì tạo bản trùng.
    """
    raw = f"{document_key}|{page}|{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    document_name: str,
    document_number: str,
    topic: str,
    status: str,
    source_url: str,
) -> Dict[str, int]:
    """Đọc PDF, chia chunk, tạo embedding và lưu vào ChromaDB."""
    pages = extract_pdf_pages(pdf_bytes)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    empty_pages = 0
    document_key = document_number.strip() or filename

    # Xóa dữ liệu cũ của cùng văn bản trước khi nạp lại.
    try:
        collection.delete(where={"document_key": document_key})
    except Exception:
        pass

    for page_number, page_text in pages:
        if len(page_text) < 50:
            empty_pages += 1
            continue

        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks):
            enriched_chunk = (
                f"Tên văn bản: {document_name}\n"
                f"Số hiệu: {document_number}\n"
                f"Chủ đề: {topic}\n"
                f"Trạng thái hiệu lực: {status}\n"
                f"Trang: {page_number}\n\n"
                f"{chunk}"
            )

            documents.append(enriched_chunk)

            metadatas.append(
                {
                    "document_key": document_key,
                    "filename": filename,
                    "document_name": document_name,
                    "document_number": document_number,
                    "topic": topic,
                    "status": status,
                    "page": page_number,
                    "source_url": source_url or "",
                }
            )

            ids.append(
                make_chunk_id(
                    document_key=document_key,
                    page=page_number,
                    chunk_index=chunk_index,
                )
            )

    if not documents:
        return {
            "pages": len(pages),
            "chunks": 0,
            "empty_pages": empty_pages,
        }

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {
        "pages": len(pages),
        "chunks": len(documents),
        "empty_pages": empty_pages,
    }


# ============================================================
# 4. TRUY XUẤT VÀ SINH CÂU TRẢ LỜI
# ============================================================

def retrieve_context(
    question: str,
    top_k: int = 5,
    active_only: bool = True,
) -> List[Dict[str, Any]]:
    """Tạo embedding câu hỏi và lấy các đoạn liên quan nhất."""
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    safe_top_k = max(1, min(top_k, total_chunks))

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [question_embedding],
        "n_results": safe_top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if active_only:
        query_kwargs["where"] = {"status": "Còn hiệu lực"}

    try:
        result = collection.query(**query_kwargs)
    except Exception:
        logger.exception("Không thể truy xuất dữ liệu từ ChromaDB")
        return []

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    distances = result.get("distances") or []

    if not documents or not documents[0]:
        return []

    items: List[Dict[str, Any]] = []

    for doc, meta, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        items.append(
            {
                "document": doc,
                "metadata": meta,
                "similarity": 1 - float(distance),
            }
        )

    return items


SYSTEM_PROMPT = """
Bạn là trợ lý tra cứu chính sách thuế Việt Nam dành riêng cho doanh nghiệp vừa và nhỏ.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên các nguồn được cung cấp trong phần NGỮ CẢNH.
2. Không tự tạo số hiệu văn bản, điều, khoản, ngày tháng, mức thuế hoặc thời hạn.
3. Nếu ngữ cảnh không đủ để kết luận, phải nói rõ:
   "Chưa đủ căn cứ trong kho dữ liệu để kết luận."
4. Dùng cách diễn đạt đơn giản, thực tế, phù hợp với chủ doanh nghiệp vừa và nhỏ.
5. Mỗi nhận định pháp lý quan trọng phải gắn với [Nguồn 1], [Nguồn 2]...
6. Nếu các nguồn mâu thuẫn hoặc khác thời điểm áp dụng, phải nêu rõ.
7. Không đưa ra hướng dẫn trốn thuế, che giấu doanh thu hoặc vi phạm pháp luật.
8. Câu trả lời chỉ mang tính hỗ trợ tra cứu, không thay thế tư vấn của cơ quan thuế hoặc chuyên gia.

CẤU TRÚC CÂU TRẢ LỜI:
- Kết luận ngắn gọn
- Giải thích áp dụng cho doanh nghiệp vừa và nhỏ
- Căn cứ được sử dụng
- Lưu ý hoặc ngoại lệ
"""


def build_context(retrieved_items: List[Dict[str, Any]]) -> str:
    """Chuyển kết quả truy xuất thành context có đánh số nguồn."""
    blocks: List[str] = []

    for index, item in enumerate(retrieved_items, start=1):
        meta = item["metadata"]

        blocks.append(
            f"""
[NGUỒN {index}]
Tên văn bản: {meta.get('document_name', '')}
Số hiệu: {meta.get('document_number', '')}
Chủ đề: {meta.get('topic', '')}
Trạng thái: {meta.get('status', '')}
Trang: {meta.get('page', '')}
URL: {meta.get('source_url', '')}

Nội dung:
{item['document']}
""".strip()
        )

    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    retrieved_items: List[Dict[str, Any]],
) -> str:
    """Gửi câu hỏi và context tới Gemini, có xử lý lỗi an toàn."""
    if gemini_client is None:
        return (
            "Hệ thống chưa được cấu hình Gemini API key. "
            "Quản trị viên cần kiểm tra phần Secrets."
        )

    context = build_context(retrieved_items)

    prompt = f"""
{SYSTEM_PROMPT}

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

NGỮ CẢNH:
{context}

Hãy trả lời bằng tiếng Việt, không sử dụng kiến thức ngoài NGỮ CẢNH.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer_text = getattr(response, "text", None)

        if not answer_text:
            return "Mô hình không trả về nội dung."

        return answer_text

    except Exception:
        logger.exception(
            "Không thể gọi Gemini API với model %s",
            GEMINI_MODEL,
        )

        return (
            "Hệ thống tạm thời không thể tạo câu trả lời. "
            "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        )


# ============================================================
# 5. GIAO DIỆN QUẢN TRỊ
# ============================================================

def render_admin_panel() -> None:
    """Hiển thị chức năng nạp và xóa dữ liệu sau khi xác thực."""
    st.subheader("Nạp văn bản thuế vào kho dữ liệu")

    uploaded_file = st.file_uploader(
        "Chọn một file PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="admin_pdf_uploader",
    )

    document_name = st.text_input(
        "Tên văn bản",
        placeholder="Ví dụ: Nghị định quy định về hóa đơn, chứng từ",
        key="admin_document_name",
    )

    document_number = st.text_input(
        "Số hiệu văn bản",
        placeholder="Ví dụ: 123/2020/NĐ-CP",
        key="admin_document_number",
    )

    topic = st.selectbox(
        "Chủ đề",
        [
            "Thuế giá trị gia tăng",
            "Hóa đơn điện tử",
            "Thủ tục khai và nộp thuế",
        ],
        key="admin_topic",
    )

    status = st.selectbox(
        "Trạng thái hiệu lực",
        [
            "Còn hiệu lực",
            "Hết hiệu lực",
            "Chưa xác minh",
        ],
        key="admin_status",
    )

    source_url = st.text_input(
        "Đường dẫn nguồn chính thức",
        placeholder="https://...",
        key="admin_source_url",
    )

    if st.button(
        "Xử lý và lưu vào kho",
        type="primary",
        key="admin_ingest_button",
    ):
        if uploaded_file is None:
            st.error("Bạn chưa chọn file PDF.")

        elif not document_name.strip():
            st.error("Bạn cần nhập tên văn bản.")

        elif not document_number.strip():
            st.error("Bạn cần nhập số hiệu văn bản.")

        else:
            try:
                with st.spinner(
                    "Đang đọc PDF, chia đoạn và tạo embedding..."
                ):
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
                        "File có thể là PDF scan. "
                        "MVP hiện tại chưa tích hợp OCR."
                    )
                else:
                    st.success(
                        f"Đã xử lý {result['pages']} trang, "
                        f"tạo {result['chunks']} đoạn dữ liệu."
                    )

                    if result["empty_pages"] > 0:
                        st.warning(
                            f"Có {result['empty_pages']} trang "
                            "không trích xuất được chữ. "
                            "Đây có thể là trang scan hoặc hình ảnh."
                        )

            except Exception:
                logger.exception("Không thể xử lý tài liệu PDF")
                st.error(
                    "Không thể xử lý tài liệu. "
                    "Hãy kiểm tra định dạng PDF và thử lại."
                )

    st.divider()
    st.subheader("Xóa dữ liệu thử nghiệm")

    confirm_delete = st.checkbox(
        "Tôi xác nhận muốn xóa toàn bộ kho dữ liệu",
        key="admin_confirm_delete",
    )

    if st.button(
        "Xóa toàn bộ dữ liệu",
        disabled=not confirm_delete,
        key="admin_delete_button",
    ):
        try:
            existing = collection.get()
            ids = existing.get("ids", [])

            if ids:
                collection.delete(ids=ids)

            st.session_state.messages = []
            st.success("Đã xóa toàn bộ dữ liệu.")

        except Exception:
            logger.exception("Không thể xóa dữ liệu ChromaDB")
            st.error(
                "Không thể xóa dữ liệu. "
                "Vui lòng kiểm tra log hệ thống."
            )


# ============================================================
# 6. GIAO DIỆN CHÍNH
# ============================================================

st.title("TaxRAG VN")
st.caption(
    "Trợ lý tra cứu chính sách thuế Việt Nam dành riêng "
    "cho doanh nghiệp vừa và nhỏ."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_admin, tab_guide = st.tabs(
    ["Hỏi đáp", "Quản trị dữ liệu", "Hướng dẫn MVP"]
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
            min_value=0.10,
            max_value=0.80,
            value=0.28,
            step=0.01,
            help=(
                "Nếu nguồn tốt nhất thấp hơn ngưỡng này, "
                "hệ thống sẽ từ chối trả lời."
            ),
        )

        st.metric("Số đoạn trong kho", collection.count())
        st.caption(f"Model sinh câu trả lời: `{GEMINI_MODEL}`")

        st.info(
            "Hệ thống chỉ hỗ trợ tra cứu cho doanh nghiệp vừa và nhỏ "
            "trong phạm vi dữ liệu đã nạp."
        )

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
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                retrieved_items: List[Dict[str, Any]] = []

                if collection.count() == 0:
                    answer = (
                        "Kho dữ liệu đang trống. "
                        "Quản trị viên cần tải văn bản thuế vào hệ thống."
                    )
                    st.warning(answer)

                else:
                    with st.spinner(
                        "Đang tìm căn cứ và tạo câu trả lời..."
                    ):
                        retrieved_items = retrieve_context(
                            question=question,
                            top_k=top_k,
                            active_only=active_only,
                        )

                        best_similarity = (
                            retrieved_items[0]["similarity"]
                            if retrieved_items
                            else 0.0
                        )

                        if (
                            not retrieved_items
                            or best_similarity < min_similarity
                        ):
                            answer = (
                                "Chưa đủ căn cứ trong kho dữ liệu để kết luận. "
                                "Hãy diễn đạt câu hỏi cụ thể hơn hoặc bổ sung "
                                "văn bản phù hợp."
                            )
                            st.warning(answer)

                        else:
                            answer = generate_answer(
                                question=question,
                                retrieved_items=retrieved_items,
                            )
                            st.markdown(answer)

                source_lines: List[str] = []

                for idx, item in enumerate(
                    retrieved_items,
                    start=1,
                ):
                    meta = item["metadata"]

                    line = (
                        f"**Nguồn {idx}:** "
                        f"{meta.get('document_name', '')} — "
                        f"{meta.get('document_number', '')} — "
                        f"trang {meta.get('page', '')} — "
                        f"độ tương đồng {item['similarity']:.2f}"
                    )

                    if meta.get("source_url"):
                        line += (
                            f" — [Mở nguồn]({meta['source_url']})"
                        )

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

    if not ADMIN_PASSWORD:
        st.error(
            "Chưa cấu hình ADMIN_PASSWORD. "
            "Quản trị viên cần thêm mật khẩu trong Secrets."
        )

    else:
        admin_password = st.text_input(
            "Mật khẩu quản trị",
            type="password",
            key="admin_password_input",
        )

        if admin_password == ADMIN_PASSWORD:
            st.success("Đã xác thực quyền quản trị.")
            render_admin_panel()

        elif admin_password:
            st.error("Mật khẩu quản trị không đúng.")

        else:
            st.info(
                "Nhập mật khẩu để sử dụng chức năng "
                "tải và quản lý tài liệu."
            )


with tab_guide:
    st.markdown(
        """
### Phạm vi MVP

MVP này dành riêng cho **doanh nghiệp vừa và nhỏ** và xử lý ba nhóm:

1. Thuế giá trị gia tăng.
2. Hóa đơn điện tử.
3. Thủ tục khai và nộp thuế.

### Quy trình quản trị

1. Mở tab **Quản trị dữ liệu**.
2. Nhập mật khẩu quản trị.
3. Tải từng PDF có lớp chữ.
4. Nhập tên, số hiệu, chủ đề, trạng thái hiệu lực và URL nguồn.
5. Nhấn **Xử lý và lưu vào kho**.
6. Quay lại tab **Hỏi đáp** để kiểm thử.

### Giới hạn của phiên bản này

- Chưa OCR PDF scan.
- Chưa tách cấu trúc Điều/Khoản hoàn chỉnh.
- Chưa có hybrid search và reranking.
- Chưa đánh giá tự động.
- ChromaDB đang được lưu cục bộ.
- Không thay thế tư vấn chính thức của cơ quan thuế hoặc chuyên gia.
"""
    )
