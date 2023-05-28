#pragma once

#include <string>

namespace tars {

enum class ErrorCode {
  SUCCESS = 0,
  ERROR = 1,
  FATAL = 2,
  UNIMPLEMENTED = 3,
};

class Status {
 public:
  Status() : error_code_(ErrorCode::SUCCESS), message_("") {}

  Status(ErrorCode ret) : error_code_(ret), message_("") {}

  Status(ErrorCode ret, const char* err_msg = "Not known")
      : error_code_(ret), message_(err_msg) {}

  static Status OK(const char* msg = "") {
    return Status{ErrorCode::SUCCESS, msg};
  }

  static Status ERROR(const char* msg = "Not known") {
    return Status{ErrorCode::ERROR, msg};
  }

  static Status FATAL(const char* msg = "succeessfully exit") {
    return Status{ErrorCode::FATAL, msg};
  }

  static Status UNIMPLEMENTED(const char* msg = "") {
    return Status{ErrorCode::SUCCESS, msg};
  }

  operator bool() const { return (error_code_ == ErrorCode::SUCCESS); }

  const char* info() const { return message_.c_str(); }

  bool operator==(const Status& status);
  bool operator!=(const Status& status);

  /// copy and move
  Status(const Status& status);
  Status(const Status&& status);
  Status& operator=(const Status& status);
  Status& operator=(const Status&& status);

 private:
  std::string message_;
  ErrorCode error_code_ = ErrorCode::SUCCESS;
};

inline bool Status::operator==(const Status& status) {
  return (this->error_code_ == status.error_code_);
}

inline bool Status::operator!=(const Status& status) {
  return (this->error_code_ != status.error_code_);
}

inline Status::Status(const Status& status) {
  this->message_ = status.message_;
  this->error_code_ = status.error_code_;
}

inline Status::Status(const Status&& status) {
  this->message_ = std::move(status.message_);
  this->error_code_ = status.error_code_;
}

inline Status& Status::operator=(const Status& status) {
  this->message_ = status.message_;
  this->error_code_ = status.error_code_;
  return *(this);
}

inline Status& Status::operator=(const Status&& status) {
  this->message_ = std::move(status.message_);
  this->error_code_ = status.error_code_;
  return *(this);
}

}  // namespace tars