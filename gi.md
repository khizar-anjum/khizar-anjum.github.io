---
layout: page
permalink: /gi.html
title: Generally Intelligent
---

Generally Intelligent (GI) is a blog that I run where I share the latest developments in the field of Artificial General Intelligence (AGI). I feel that the current popular emphasis of Artificial Intelligence (AI) on buying performance by using unlimited data and fragmentation of pursuit of intelligence into several small and highly specific subtasks is misleading at best. This blog is meant to start the discussion on AGI in a more informal way. 

<section>
	<div class="posts">
		{% assign sorted_posts = site.gi | sort:"date" | reverse %}
		{% for post in sorted_posts %}
			<article>
				<a href="{{post.permalink}}" class="image"><img src="{{post.poster}}" alt="" /></a>
				<h3> {{post.title}} </h3>
				<p> {{post.summary}} </p>
				<ul class="actions">
					<li><a href="{{post.permalink}}" class="button">More</a></li>
				</ul>
			</article>
		{% endfor %}
	</div>
</section>