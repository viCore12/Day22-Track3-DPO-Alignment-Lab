# Customer-service Chatbot: Hanoi Coffee Roastery

Dự án này là kết quả của Bonus Challenge (Provocation #2) nhằm áp dụng DPO vào việc xây dựng một chatbot chăm sóc khách hàng có "hồn" cho một doanh nghiệp thực tế.

## 1. Audience (Đối tượng sử dụng)
Khách hàng của quán cà phê "Hanoi Coffee Roastery" (một quán cà phê specialty giả định tại Hà Nội) nhắn tin qua Fanpage hoặc Website để hỏi về menu, giờ mở cửa, cách đặt bàn, và khiếu nại dịch vụ.

## 2. Domain Knowledge (Kiến thức nghiệp vụ)
Chatbot được huấn luyện để hiểu và tuân thủ các quy tắc sau:
- **Tone & Voice:** Lịch sự, thân thiện, mang phong cách phục vụ quán cà phê trẻ trung (xưng "quán em", "mình", "dạ").
- **Kiến thức Menu:** Quán chuyên phục vụ Specialty Coffee (Pour over, Cold Brew) và các loại trà thanh nhiệt. Không phục vụ sinh tố, trà sữa hay rượu.
- **Giờ hoạt động:** 7:00 AM - 10:30 PM mỗi ngày.
- **Xử lý tình huống:** Luôn có Call-to-action (CTA) rõ ràng (ví dụ: "Mình đi mấy người để em xếp bàn ạ?"). Tuyệt đối không bịa thông tin ngoài menu hoặc hứa hẹn những điều quán không làm được.

## 3. Application Objective (Mục tiêu ứng dụng)
Biến một model base ngôn ngữ chung (thường trả lời kiểu Wikipedia hoặc kiểu ChatGPT khô khan) thành một nhân viên chăm sóc khách hàng trực tuyến:
- Trả lời nhanh gọn, đi thẳng vào vấn đề.
- Có khả năng từ chối khéo léo (vd: khi khách hỏi trà sữa).
- Tăng tỷ lệ chuyển đổi bằng cách hỏi ngược lại khách hàng (CTA).

## 4. Real-world Output (Sản phẩm bàn giao)
- **Data:** Tập dataset `prompts.jsonl` và script `generate_data.py` tự tạo dữ liệu preference (`pairs.parquet`).
- **Training:** Script huấn luyện DPO `train.py` chạy trên Llama/Qwen.
- **Deploy:** File `serve.py` cung cấp giao diện Gradio và FastAPI để quán có thể tích hợp vào hệ thống hiện tại.
- **Documentation:** `MODEL-CARD.md` liệt kê rõ giới hạn của model.

---

## 5 Sample Interactions (Ví dụ tương tác)

Xem chi tiết 5 ví dụ so sánh sự khác biệt của model trước và sau khi DPO tại: [`demo/5-samples.md`](demo/5-samples.md)

| Prompt | Generic SFT/Base Output | DPO Chatbot Output |
|--------|--------------------------|---------------------|
| Khách: Quán có trà sữa trân châu không? | Xin lỗi, tôi là AI nên không có thông tin về menu hiện tại của quán. Bạn vui lòng liên hệ hotline nhé. | Dạ quán em chuyên về cà phê ủ lạnh và các loại trà trái cây thanh mát, hiện quán không phục vụ trà sữa ạ. Mình có muốn thử món Trà Vải Hoa Hồng đang best-seller bên em không? |

## Limitations & Future Work
- **What this POC doesn't handle yet:** Chưa kết nối với database thời gian thực để check tình trạng bàn trống hay món hết.
- Model đôi khi vẫn có thể "ảo giác" nếu khách hỏi về những món ăn kèm quá lạ chưa từng xuất hiện trong tập huấn luyện.
