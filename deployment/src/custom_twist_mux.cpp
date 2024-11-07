#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

class TwistMux : public rclcpp::Node
{
public:
    TwistMux(const std::string& config_file_path)
        : Node("twist_mux")
    {
        if (!loadConfig(config_file_path)) {
            throw std::runtime_error("Config file error");
        }

        output_topic_ = config_["twist_mux"]["output"].as<std::string>();
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(output_topic_, 10);

        for (const auto& topic : config_["twist_mux"]["topics"]) {
            setupTopic(topic);
        }
    }

private:
    bool loadConfig(const std::string& config_file_path)
    {
        try {
            config_ = YAML::LoadFile(config_file_path);
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading config file: %s", e.what());
            return false;
        }
    }

    void setupTopic(const YAML::Node& topic)
    {
        std::string topic_name = topic["topic"].as<std::string>();
        double timeout = topic["timeout"].as<double>();
        int priority = topic["priority"].as<int>();

        RCLCPP_INFO(this->get_logger(), "Subscribing to topic: %s with priority %d and timeout %.1f",
                    topic_name.c_str(), priority, timeout);

        auto callback = [this, topic_name, priority, timeout](const geometry_msgs::msg::Twist::SharedPtr msg) {
            this->twistCallback(msg, topic_name, priority, timeout);
        };

        subscriptions_[topic_name] = this->create_subscription<geometry_msgs::msg::Twist>(topic_name, 10, callback);
        topic_timeouts_[topic_name] = timeout;
        topic_priorities_[topic_name] = priority;
    }

    void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg, const std::string& topic_name, int priority, double timeout)
    {
        auto now = this->now();
        latest_messages_[topic_name] = *msg;
        last_received_time_[topic_name] = now;

        RCLCPP_INFO(this->get_logger(), "Received message from topic: %s", topic_name.c_str());

        updateActiveTopics();
        publishHighestPriorityMessage();
    }

    void updateActiveTopics()
    {
        auto now = this->now();
        active_topics_.clear();

        for (const auto& [topic, last_time] : last_received_time_) {
            if ((now - last_time).seconds() <= topic_timeouts_[topic]) {
                active_topics_.insert(topic);
            }
        }
    }

    void publishHighestPriorityMessage()
    {
        std::string selected_topic;
        int highest_priority = -1;

        for (const auto& topic : active_topics_) {
            int priority = topic_priorities_[topic];
            if (priority > highest_priority) {
                highest_priority = priority;
                selected_topic = topic;
            }
        }

        if (!selected_topic.empty()) {
            publisher_->publish(latest_messages_[selected_topic]);
            RCLCPP_INFO(this->get_logger(), 
                "Publishing message from topic: %s (priority: %d)", 
                selected_topic.c_str(), 
                topic_priorities_[selected_topic]);
            
            std::string active_topics_str = "Active topics: ";
            for (const auto& topic : active_topics_) {
                active_topics_str += topic + ", ";
            }
            RCLCPP_INFO(this->get_logger(), "%s", active_topics_str.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "No topic selected for publishing");
        }
    }

    YAML::Node config_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    std::unordered_map<std::string, rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr> subscriptions_;
    std::string output_topic_;
    std::unordered_map<std::string, geometry_msgs::msg::Twist> latest_messages_;
    std::unordered_map<std::string, rclcpp::Time> last_received_time_;
    std::unordered_map<std::string, double> topic_timeouts_;
    std::unordered_map<std::string, int> topic_priorities_;
    std::unordered_set<std::string> active_topics_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: ros2 run twist_mux twist_mux --config <path_to_config_file>" << std::endl;
        return 1;
    }

    std::string config_file_path = argv[1];

    try {
        auto node = std::make_shared<TwistMux>(config_file_path);
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TwistMux node: " << e.what() << std::endl;
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}