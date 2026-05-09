---
language: vi
license: mit
tags:
- dpo
- alignment
- customer-service
- chatbot
- vietnamese
---

# Model Card cho Hanoi Coffee Roastery Chatbot

## Model Details
- **Developer:** Nhanvi282 (Lab 22 Bonus Challenge)
- **Base Model:** `unsloth/Qwen2.5-3B-bnb-4bit` (hoặc model tương tự được huấn luyện qua SFT)
- **Alignment Method:** Direct Preference Optimization (DPO)
- **Language:** Tiếng Việt

## Intended Use
Model được huấn luyện đặc biệt cho mục đích làm **Chatbot Chăm sóc Khách hàng** cho quán cà phê "Hanoi Coffee Roastery". 
- **Người dùng mục tiêu:** Khách hàng nhắn tin trên Fanpage/Zalo/Website của quán.
- **Mục tiêu cốt lõi:** Trả lời tự nhiên, thân thiện (xưng "dạ", "em", "mình"), cung cấp thông tin chính xác về giờ mở cửa, menu, cách đặt bàn, và dẫn dắt khách hàng đưa ra quyết định (Call-to-action).

## Out-of-Scope Use
Đây là **DOMAIN-SPECIFIC MODEL**, vì vậy nó có giới hạn rất chặt chẽ:
- **KHÔNG** dùng để tư vấn sức khoẻ, pháp lý hay các lĩnh vực chuyên môn khác.
- **KHÔNG** nên dùng làm Chatbot cho các brand có phong cách nói chuyện khác biệt (vd: ngân hàng cần sự trang trọng tuyệt đối, hoặc genZ năng động dùng nhiều tiếng lóng).
- **KHÔNG** yêu cầu model tính toán hóa đơn phức tạp có nhiều mã giảm giá (dễ hallucinate số liệu).

## Training Data
Mô hình được huấn luyện DPO với 100-200 cặp preference:
- `chosen`: Văn phong lịch sự, đúng sự thật, có CTA.
- `rejected`: Văn phong robot (ChatGPT-style), hoặc trả lời dài dòng không trọng tâm, bịa món ăn.

## Known Limitations
1. **Real-time Data:** Model không có quyền truy cập vào hệ thống quản lý kho, do đó nó không biết hôm nay quán có hết hạt cà phê Ethiopia hay không. (Cần kết hợp với RAG hoặc Function Calling để khắc phục).
2. **Hallucination:** Dù đã qua DPO để giảm thiểu, model đôi lúc vẫn có thể bịa ra một mức giá nếu khách hàng hỏi ép. 
3. **Context Length:** Được tối ưu hóa cho các đoạn chat ngắn gọn gọn (<200 từ), sẽ hoạt động kém nếu khách hàng dán một đoạn review dài 1000 từ.

## AI Workflow Vibe Coding
*Ghi chú về việc sử dụng AI trong phát triển:*
Prompt tạo dataset hiệu quả nhất: "Đóng vai khách hàng quán cà phê hỏi những câu hóc búa nhất về menu, sau đó tạo 1 cặp trả lời: 1 câu như nhân viên quán, 1 câu như AI vô hồn."
Prompt thất bại: "Viết script DPO tự động hóa mọi thứ" (Thiếu chi tiết về format thư viện Unsloth dẫn đến code không chạy).
