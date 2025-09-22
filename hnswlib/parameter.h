//
// Created by jlc on 9/18/25.
//

#ifndef PARAMETER_H
#define PARAMETER_H
#include <sstream>
#include <typeinfo>
#include <unordered_map>

namespace hnswlib
{
    struct reverseNN_info {
        tableint id;                  // id值
        unsigned length;            // id的reverseNN树目（即rNNs.size()）
        std::vector<tableint> rNNs;   // id的对应rNNs的id值

        reverseNN_info() = default;
        reverseNN_info(tableint id)
            : id{id}, length{0}, rNNs{} {}
        reverseNN_info(tableint id, unsigned reserve_size)
            : id{id}, length{0}, rNNs{} {rNNs.reserve(reserve_size);}
        reverseNN_info(tableint id, unsigned length, std::vector<tableint> rNNs)
            : id{id}, length{length}, rNNs{std::move(rNNs)} {}
    };

    struct block_info {
        tableint bid;
        std::vector<tableint> bmembers;

        block_info() = default;
        block_info(tableint bid)
            : bid{bid}, bmembers{} {}
        block_info(tableint bid, unsigned reserve_size)
            : bid{bid}, bmembers{} {bmembers.reserve(reserve_size);}
        block_info(tableint bid, std::vector<tableint> bmembers)
            : bid{bid}, bmembers{std::move(bmembers)} {}
    };

    class Parameters {
    public:
        template <typename ParamType>
        inline void Set(const std::string &name, const ParamType &value) {
            std::stringstream sstream;
            sstream << value;
            params[name] = sstream.str();
        }

        // NOTE: pybind11 do not support template functions.
        // Since add auto type convertion is not trival,
        // we just return string directly as a work around.
        inline std::string GetRaw(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid parameter name.");
            } else {
                return item->second;
            }
        }

        template <typename ParamType>
        inline ParamType Get(const std::string &name) const {
            auto item = params.find(name);
            if (item == params.end()) {
                throw std::invalid_argument("Invalid parameter name.");
            } else {
                return ConvertStrToValue<ParamType>(item->second);
            }
        }

        template <typename ParamType>
        inline ParamType Get(const std::string &name,
                             const ParamType &default_value) {
            try {
                return Get<ParamType>(name);
            } catch (std::invalid_argument e) {
                return default_value;
            }
        }

    private:
        std::unordered_map<std::string, std::string> params;

        template <typename ParamType>
        inline ParamType ConvertStrToValue(const std::string &str) const {
            std::stringstream sstream(str);
            ParamType value;
            if (!(sstream >> value) || !sstream.eof()) {
                std::stringstream err;
                err << "Failed to convert value '" << str
                    << "' to type: " << typeid(value).name();
                throw std::runtime_error(err.str());
            }
            return value;
        }
    };
}
#endif //PARAMETER_H
