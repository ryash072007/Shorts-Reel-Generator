from research import YouTubeRedditResearcher
import config
import os

def main():
    """Run the YouTube Shorts research tool with configured settings."""
    print("YouTube Shorts Research Tool")
    print("----------------------------")
    
    # Initialize the researcher
    researcher = YouTubeRedditResearcher()
    
    # Conduct niche research if configured
    if config.MODE in ['niche', 'all']:
        print(f"Searching for: {' '.join(config.SEARCH_KEYWORDS)}")
        researcher.search_reddit_shorts(
            query_terms=config.SEARCH_KEYWORDS,
            max_results=config.MAX_RESULTS
        )
        
        # Basic analysis
        print("Performing basic content analysis...")
        researcher.analyze_trending_subreddits()
        researcher.analyze_content_trends()
        untapped_niches = researcher.identify_untapped_niches()
        
        # Enhanced analysis
        print(f"Analyzing growth patterns across {config.MAX_CHANNELS} channels...")
        researcher.analyze_channel_growth(max_channels=config.MAX_CHANNELS)
        researcher.analyze_success_patterns()
        researcher.create_competitive_landscape()
        researcher.analyze_content_evolution()
        
        # Content gap analysis
        print("Performing content gap analysis...")
        gap_analysis = researcher.perform_content_gap_analysis()
        researcher.gap_analysis = gap_analysis  # Store for Groq to analyze
        
        # Generate reports
        print("Generating research reports...")
        report = researcher.generate_report()
        enhanced_report = researcher.generate_enhanced_report()
        
        # Save research data
        output_file = f"{config.OUTPUT_BASE_FILENAME}.json"
        researcher.save_enhanced_research(output_file)
        
        # Generate visualizations if requested
        if config.GENERATE_VISUALIZATIONS:
            print("Generating visualizations...")
            researcher.plot_top_performers()
            researcher.visualize_content_trends()
        
        # Write reports to files
        with open(f"{config.OUTPUT_BASE_FILENAME}_basic_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        with open(f"{config.OUTPUT_BASE_FILENAME}_enhanced_report.md", "w", encoding="utf-8") as f:
            f.write(enhanced_report)
            
        print(f"Basic report saved to {config.OUTPUT_BASE_FILENAME}_basic_report.md")
        print(f"Enhanced report saved to {config.OUTPUT_BASE_FILENAME}_enhanced_report.md")
        
        # Always perform Groq analysis for niche research with specialized prompt
        print("Performing advanced AI analysis with Groq for niche insights...")
        groq_analysis = researcher.analyze_with_groq(analysis_type="niche")
        if groq_analysis:
            with open(f"{config.OUTPUT_BASE_FILENAME}_niche_groq_analysis.md", "w", encoding="utf-8") as f:
                f.write(groq_analysis)
            print(f"Groq niche analysis saved to {config.OUTPUT_BASE_FILENAME}_niche_groq_analysis.md")
    
    # Conduct channel analysis if configured
    if config.MODE in ['channel', 'all'] and config.CHANNEL_ID:
        print("Analyzing your YouTube channel...")
        channel_analysis = researcher.analyze_your_channel(channel_id=config.CHANNEL_ID)
        
        if channel_analysis:
            channel_report = researcher.generate_channel_report(channel_analysis)
            with open(f"{config.OUTPUT_BASE_FILENAME}_channel_report.md", "w", encoding="utf-8") as f:
                f.write(channel_report)
            print(f"Channel report saved to {config.OUTPUT_BASE_FILENAME}_channel_report.md")
            
            # Always use Groq for deep channel analysis
            print("Performing advanced AI analysis with Groq for channel success factors...")
            channel_groq_analysis = researcher.analyze_channel_with_groq(channel_analysis)
            if channel_groq_analysis:
                with open(f"{config.OUTPUT_BASE_FILENAME}_channel_success_factors.md", "w", encoding="utf-8") as f:
                    f.write(channel_groq_analysis)
                print(f"Groq channel analysis saved to {config.OUTPUT_BASE_FILENAME}_channel_success_factors.md")

if __name__ == "__main__":
    main()
